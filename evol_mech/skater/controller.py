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

from ..body.hardware import (tread_drive_from_params, servo_from_params, roll_servo_from_params, fan_from_params,
                             SERVO_LATENCY_S, SERVO_CMD_HZ, SERVO_CMD_DEADBAND_RAD, SERVO_MIN_SPEED_FRAC)
from ..sim.controller import quat_to_roll_pitch
from .mjcf import ACTUATOR_NAMES, FAN_PITCH_RANGE
from .sensors import FootSensors
from .structure import (nominal_angles, nominal_height, nominal_foot, angles_from_foot,
                        horn_from_shin, shin_from_horn, drive_stops, KNEE_RANGE, hip_roll_range,
                        roll_ratio, roll_arm, is_pantograph, panto_nominal, panto_ik, panto_pose_allowed, ROLL_PIVOT_Y,
                        panto_reach)

MU_FOOT = 1.2
G = 9.81
A_MAX = 4.0                 # m/s^2 cap on demanded body acceleration
TIP_MARGIN = 0.5            # treads may use this fraction of the tip-over acceleration
COM_ABOVE_HIP = 0.08        # rough centre-of-mass height above the hip joint (core + battery)
TREAD_TRACK_GAIN = 10.0     # 1/s, angle-error term of the tread position hold
AIRBORNE_Z = 0.03           # hip higher than nominal by this -> treat as airborne
FAN_AUTH_THR = 0.1          # fan pitch feedback has full authority from this mean throttle up
FLIGHT_PUSH_S = 0.1         # legged: the fans push forward only after this long with no foot down (a jump, not a jog's gap)
LEVEL_SOLES = True          # sole-levelling parallelogram at the ankle (see SkaterController._level_soles)
RUN_RAMP_S = 0.6            # run: the gait amplitudes (sweep, crouch, lift) ease in over this long from the parked stance
CONTROL_DT = 0.01
LEGGED = ("walk", "run", "course", "hop", "broad_jump")      # objectives with locked treads and moving feet
# course: a schedule of gaits. The commanded gait sets the targets of a stepping amplitude
# (0 standing, 1 stepping) and of the walk/run mix; both slew over the trans_s gene, one shared phase
# runs at the blended cadence, and the two cycles are blended at that phase.
COURSE_PROFILE = (("stand", 2.0), ("walk", 10.0), ("run", 10.0), ("walk", 10.0), ("stand", 4.0))
# walk cycle: stance duty (`step_duty`) and the cycle fraction at which the body is
# furthest left (`shift_phase`) are genes; the left foot's phase is the cycle reference
# (u = 0 its touchdown, u = duty its swing). Every gait starts at u = duty, so the left
# foot takes the first step, and a course stop ends with the right foot's swing landing
# beside the left (the closing step) before the stepping amplitude fades: start and stop
# are the same every time.
JUMP_PHASES = (0.45, 0.60, 0.80)            # cycle fractions: crouch | thrust | tuck | settle
# contact-driven walk (task gait_phase "contact"; default "clock" keeps every score). The shared phase
# still runs at step_hz, but a swing foot that finds the floor early pulls the phase to its touchdown
# over CATCH_S, a foot that has not found it by its touchdown mark holds the phase for up to
# CONTACT_WAIT_S while the foot searches down at SEARCH_DOWN_M_S (then the clock takes over), and
# each stance sweep starts from where the foot's sensor says it landed (x and y), not the nominal
# landing point: a planted foot is never asked to slide. The start is a sequence: the body shift
# ramps toward the first stance foot over START_RAMP_S and the first swing waits for that foot's
# FSRs to read START_LOAD_BW (or START_TIMEOUT_S).
CATCH_S = 0.05
CONTACT_WAIT_S = 0.10
SEARCH_DOWN_M_S = 0.30
LAND_LOAD_BW = 0.25       # a landing is this much load for LAND_STEPS steps; a brush of the floor mid-swing is not
LAND_STEPS = 2
LIFT_STEPS = 3            # the swing foot must have read no contact this many steps running before it can land early (a shuffle cannot)
SEARCH_MAX_M = 0.03
SWING_MIN_FRAC = 0.4        # a landing counts only after this much of the swing (the lift-off debounce)
CATCH_ARM_S = 0.5           # the catch step is armed only after the start's settling (the first steps rock the body)
CROUCH_RATE = 0.3           # m/s the hip drops during a catch (task key catch_crouch)
LEVEL_FADE = 0.3            # level_swing: the levelling fades out over this fraction of the stance after touchdown
START_RAMP_S = 0.4
START_LOAD_BW = 0.6
START_TIMEOUT_S = 0.8
ANCHOR_FROM_MEASUREMENT = True   # False: the stance sweep and swing start from the nominal +-L/2 (a diagnostic)
HOLD_STOPS_SWEEP = False  # True: a held phase (late landing) also holds the planted foot, so the body waits; False: the sweep runs on
# the contact gait on a strider (measured 2026-11-08 on the round-1e walker, real servo): the parked start
# (shift ramp, wait for the stance foot's load) drops it inside 1.2 s, it cannot stand still on split feet; the
# measured anchors drop it too, the foot lands 4 cm behind its lagging command and a sweep started from the
# measurement walks the support out from under the body; the late hold and the early pull are fine (the hold
# even helps: the strider's feet land late on nearly every step). So two task keys, defaults keep the shuffler's
# gait: contact_start 'loaded' | 'instant' (start stepping at t = 0 like the clock gait) and contact_anchor
# 'measured' | 'command' (each stance sweep starts from the commanded landing point, +-L/2). A body-shift from
# the gait state (hip over the stance foot through the other foot's swing, full track width) was tried for a
# true single-support gait and fell at 0.75 s on every start: the lateral shift goes through the hip roll
# servos, rolls the body, and the roll-levelling term lifts the foot the body is moving onto.
# capture-point foot placement (task foot_placement "capture"; default "gains" keeps every score). Two
# balance strategies, kept apart: the planted feet shift under the body by the kp / kd gains (the
# ankle strategy: a centre-of-pressure shift within the soles, which the GA has always tuned small),
# and the SWING foot lands at the linear inverted pendulum's capture point, hip velocity x
# sqrt(h_com / g) ahead of its nominal landing (the step strategy), weighted in over the swing. The
# kv_foot_x / kv_foot_y velocity gains, whose job that was, go. Applying the full capture point to
# both planted feet at every step was tried first and fell in a second on every walker: it is a
# 40 cm/rad body translation through the roll servos against feet that do not slide. Sizing the body
# shift to the measured feet was tried too and dropped: it unsettled every clock-tuned gait, and the
# shift's gene formula already sizes it to the stance foot. Correct at any hip width by construction.
REACH_MARGIN = 0.01         # the workspace clamp keeps the ankle this far inside the leg's straight-down reach
KNEE_FROM_MEASURED_THIGH = False  # knee target: four-bar at the commanded thigh. True = at the measured thigh,
                                  # closer to a rigid link but the two position loops then chatter (rms accel x3)


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SkaterController:
    def __init__(self, model: mujoco.MjModel, p: dict[str, float], task: dict,
                 fans_enabled: bool = True, sensor_rng: np.random.Generator | None = None):
        self.p, self.task = p, task
        self.fans_enabled = fans_enabled
        self.idx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES}
        self.ctrl = np.zeros(model.nu)
        hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
        self.hip_id = hip_id
        self.mass = float(model.body_subtreemass[hip_id])
        self.tread = tread_drive_from_params(p)
        self.servo, self.roll_servo = servo_from_params(p), roll_servo_from_params(p)
        # the real servo model: speed falls with load, commands arrive late, held at 50 Hz, in deadband steps
        self.real_servo = str(task.get("servo_model", "ideal")) == "real"
        # latency compensation (task servo_lead: seconds, or "auto" = the latency plus the hold, 40 ms;
        # default 0 keeps every score): the controller commands for when the command lands. The
        # gait is evaluated servo_lead ahead and the balance terms are led by the tilt rate, which
        # on the round-9 strider is the difference between falling in 2-6 s and standing 12 s on
        # every start (the servo-model ablation in docs/skater_review.md: its 20 ms latency alone
        # was the fall). 40 ms stands four of four, 30 ms three, 50 ms one: the full hold counts
        lead = task.get("servo_lead", 0.0)
        self.lead = (SERVO_LATENCY_S + 1.0 / SERVO_CMD_HZ) if str(lead) == "auto" else float(lead)
        self.servo_acts = [self.idx[f"{j}_{s}"] for j in ("hip_roll", "hip_pitch", "knee") for s in "lr"]
        self._servo_queue: list[np.ndarray] = []      # commands issued, oldest first, for the latency
        self._servo_held: np.ndarray | None = None    # the command the servos are holding (50 Hz zero-order hold)
        self._load_frac: dict[str, float] = {}        # each servo's torque fraction last step, for the speed-torque line
        self._data = None
        self._tick = 0
        self.t_max = fan_from_params(p).max_thrust_n
        self.roller_q = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"roller_{s}0")]
                         for s in ("l", "r")}
        self.tread_ref = {"l": 0.0, "r": 0.0}       # tracked roller angle
        self.z_ref = nominal_height(p)
        self.x_nom, z_nom = nominal_foot(p)
        self.h_nom = -z_nom                          # hip-to-ankle drop in the parked crouch
        self.roll_range, self.roll_ratio, self.roll_arm = hip_roll_range(p), roll_ratio(p), roll_arm(p)
        # inboard feet (stance_w < 0) stop a sole's width plus a centimetre short of each other:
        # the soles only collide with the floor in the model, so the clamp keeps them apart
        self.stance_w = max(p["stance_w"], -(p["hip_width"] - p["tread_w"] - 0.01) / 2)
        self.panto = is_pantograph(p)
        # each leg's hip point (the roll pivot's y plus the bracket arm; the horns' midpoint on the
        # pantograph) in the hip body's frame, and the ankle target the IK was last solved for, in that
        # frame: the rollout holds the foot's actual track against it (foot_track_rms_m)
        self.hip_point = {s: np.array([0.0, sign * (ROLL_PIVOT_Y + self.roll_arm) if self.panto else sign * p["hip_width"] / 2, 0.0])
                          for s, sign in (("l", 1.0), ("r", -1.0))}
        self.foot_target = {s: self.hip_point[s] + np.array([0.0, 0.0, -nominal_height(p)]) for s in "lr"}
        qadr = lambda name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        if self.panto:
            # the pantograph: both leg servos are horns; the IK warm-starts from the last pose
            hf, hr = panto_nominal(p)
            self.ik_guess = {s: (hf, hr) for s in ("l", "r")}
            self.ik_nominal = (hf, hr)
            # the workspace clamp (task leg_reach "clamp"; default off keeps every score): the ankle
            # target is held inside the leg's reach, REACH_MARGIN short of full extension, before the
            # IK, and an IK that does not converge from the last pose is retried from the crouch.
            # Found by the phase-1 stall diagnosis: the round-8 strider ran its stance leg 1-2 cm
            # past the pantograph's reach for a third of every stance even under the ideal servo
            # (the GA had learned to lean on the IK's nearest-pose fallback), and under the real
            # servo the fallback wandered onto the other solution branch (both horns at +130 deg)
            # and stayed there: 20-50 mm of foot error, half the horn slew in chatter, a fall in 2 s
            self.reach_clamp = str(task.get("leg_reach", "free")) == "clamp"
            self.reach_max = panto_reach(p) - REACH_MARGIN
            self.servo_cmd = {f"{j}_{s}": (hf if j == "hip_pitch" else hr if j == "knee" else 0.0)
                              for s in ("l", "r") for j in ("hip_roll", "hip_pitch", "knee")}
            # the foot hangs from link_c: its body-relative angle is the chain's qpos sum
            self.chain_q = {s: [qadr(f"hip_pitch_{s}"), qadr(f"rod_f_{s}"), qadr(f"link_c_{s}")] for s in ("l", "r")}
        else:
            thigh, shin = nominal_angles(p)
            # the shin drive: horn hard stops per thigh angle, and the horn that parks the shin
            self.stops = drive_stops(p)
            self.thigh_lim = self.stops.thigh_range()
            horn = horn_from_shin(p, thigh, shin)
            self.horn_nom = thigh + p["crouch"] if horn is None else horn
            self.servo_cmd = {f"{j}_{s}": (thigh if j == "hip_pitch" else self.horn_nom if j == "knee" else 0.0)
                              for s in ("l", "r") for j in ("hip_roll", "hip_pitch", "knee")}
            self.chain_q = {s: [qadr(f"hip_pitch_{s}"), qadr(f"knee_{s}")] for s in ("l", "r")}
        self.thigh_q = {s: qadr(f"hip_pitch_{s}") for s in ("l", "r")}
        # sole-levelling parallelogram: the ankle spring's rest follows the leg so the sole
        # stays parallel to the body at any knee bend (see _level_soles)
        self.model = model
        self.ankle_q = {s: qadr(f"ankle_{s}") for s in ("l", "r")}
        reach = max(0.005, p["sole_len"] / 2 - p["pulley_radius"])
        self.a_tip = TIP_MARGIN * G * reach / (self.z_ref + COM_ABOVE_HIP)
        self.objective = task["objective"]
        self.legged = self.objective in LEGGED
        self.speed = float(task.get("speed", 1.0))
        self.p0 = None
        self.last_throttle = (0.0, 0.0)
        self.last_fan_force = (0.0, 0.0)
        self.x_int = 0.0                             # legged: integrated foot shift (steady moments)
        self.horn_cmd = {"l": 0.0, "r": 0.0}
        self.air_since = None                        # legged: time the feet left the floor
        # course state (see _course_update); cycles counts the shared phase for every objective
        self.profile = [(str(g), float(d)) for g, d in task.get("profile", COURSE_PROFILE)]
        self.amp, self.cycles, self.phase = 0.0, p["step_duty"], p["step_duty"]
        self.closing = None          # course stop: None | "wait" (for the left touchdown) | "swing" (braking, the right foot closing last) | "hold" (feet together, amplitude fading)
        self._brake = (0.0, 0.5)     # (cycles at the braking start, cycles the brake lasts: a whole number plus a half)
        self.blend = 1.0 if self.profile[0][0] == "run" else 0.0    # a course that opens with a run is a run from t = 0
        self._t_prev = None
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geoms = {g for g in range(model.ngeom)
                           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("foot_")}
        # the feet's force-sensing resistors, ankle angles and servo load, read every control
        # step (sensors.py); phase 0 of the test plan reads them, phase 1 drives the gait off them
        self.sensors = FootSensors(model, p, self.mass, CONTROL_DT, rng=sensor_rng)
        # contact-driven walk state (see the constants above)
        self.contact_gait = str(task.get("gait_phase", "clock")) == "contact" and self.objective == "walk"
        self.foot_body = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{s}") for s in "lr"}
        self.foot_meas = {s: np.array([0.0, 0.0, -self.h_nom]) for s in "lr"}     # ankle, hip frame, from the leg's hip point
        self.gait_dy = {"l": 0.0, "r": 0.0}        # lateral landing offset at touchdown
        self.dy_hold = {"l": 0.0, "r": 0.0}        # ... as applied now (held in stance, fading over the swing)
        self._x_bal = {"l": 0.0, "r": 0.0}         # last balance x target per foot (the gait offset's reference)
        self._dy_tgt = {"l": 0.0, "r": 0.0}        # last lateral target without the held offset
        self._anchor = {s: p["step_len"] / 2 for s in "lr"}      # stance sweep start, as a gait x offset
        self._swing_from = {s: -p["step_len"] / 2 for s in "lr"}   # where this swing left the floor, as a gait x offset
        self._landed = {"l": False, "r": False}    # this swing's landing already seen
        self._lifted = {"l": False, "r": False}    # this swing's foot has actually left the floor (a dragged foot cannot land early)
        self._off = {"l": 0, "r": 0}               # consecutive steps without contact this swing
        self._loaded = {"l": 0, "r": 0}            # consecutive steps the swing foot has read a landing's load
        self._search = {"l": 0.0, "r": 0.0}        # how far the foot is searching down for the floor
        self._wait = {"l": 0.0, "r": 0.0}          # how long the phase has waited for this foot
        self._gait_started = False
        self._tw_prev = None
        self.contact_events = {"early": 0, "late": 0, "clock": 0}      # for the rollout's report
        self.anchor_measured = ANCHOR_FROM_MEASUREMENT and str(task.get("contact_anchor", "measured")) != "command"
        self.instant_start = str(task.get("contact_start", "loaded")) == "instant"
        self.capture = str(task.get("foot_placement", "gains")) == "capture"
        self.swing_w = {"l": 0.0, "r": 0.0}        # 0 in stance, the swing's progress 0..1 in swing (walk only)
        self.measure_feet = self.contact_gait or self.capture
        # catch step (task key catch_step, clock walk only): see _catch_update
        self.catch_on = bool(task.get("catch_step", False)) and self.objective == "walk" and not self.contact_gait
        self.catch_pitch = float(task.get("catch_pitch", 0.08))     # rad of pitch away from upright ...
        self.catch_rate = float(task.get("catch_rate", 0.3))        # ... and rad/s still growing, to trigger
        self.catch_gain = float(task.get("catch_gain", 1.0))        # x the capture point (hip speed x pendulum time)
        self.catch_down = float(task.get("catch_down", 0.01))       # m below the stance foot's ground the catch reaches
        self.catch_timeout = float(task.get("catch_timeout", 0.5))  # s before a catch that has not landed gives up
        self.clock_off = 0.0                       # cycles added to the walk clock (frozen during a catch, re-phased after)
        self.catch_side, self.catch_t0, self.catch_land = None, 0.0, 0
        self.catch_x = {"l": 0.0, "r": 0.0}        # the catch foot's fore-aft target (hip frame, beside x_f) while catching
        self.x_bias = {"l": 0.0, "r": 0.0}         # after a catch: the planted foot's offset from the stance path, fading
        self.x_bias_rate = {"l": 0.0, "r": 0.0}
        self.catch_events = 0
        # the leg walks nearly straight (34 cm hip, 31 cm reach), so a foot placed behind and down at the ground is out
        # of reach: during a catch the stance leg bends catch_crouch (m), lowering the hip so the catch can land
        self.catch_crouch = float(task.get("catch_crouch", 0.0))
        self.crouch_now = 0.0
        # task key level_legs k (0 = off): the foot pattern is laid out in a frame levelled in pitch (k = 1 fully) and
        # turned into the tilted hip frame each step, so a body that pitches does not carry the whole leg pattern with
        # it (on cobbles the walkers tipped past ~15 deg, both feet ended ahead of the centre of mass in the world and
        # the robot rotated over its heels as one piece, catch step or not). Roll is left to the hip roll servos.
        self.level_legs = float(task.get("level_legs", 0.0))
        # task key level_swing k: the same levelling for the swing foot only, fading out over the first LEVEL_FADE of its
        # stance (levelling the stance leg frees the torso in pitch: a pendulum 8 cm tall, too quick for the servos)
        self.level_swing = float(task.get("level_swing", 0.0))
        self.cop_gain = float(task.get("cop_gain", 0.0))       # heel-toe pressure feedback (see act); 0 = off
        self.cop_dead = float(task.get("cop_dead", 0.4))
        self.cop_e = 0.0
        self.u_phase = {"l": 0.0, "r": 0.0}

    # ---- 1. planner ------------------------------------------------------------
    def plan(self, pos: np.ndarray, yaw: float) -> tuple[np.ndarray, float]:
        p, obj = self.p, self.objective
        if obj in ("skate", "walk", "course", "broad_jump"):
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
    @staticmethod
    def _leg(u: float, L: float, duty: float, lift: float, crouch: float, ext: float) -> tuple[float, float]:
        """One stepping cycle at phase u (0 = touchdown): (foot x offset, foot / hip
        height offset). The stance foot sweeps L back under the hip for duty of the cycle;
        the swing foot returns lifted by lift. With crouch the hip also rides crouch low
        (the bent-knee landing that loads the leg) and recovers ext of it in the last half
        of stance (the thrust): with duty < 0.5 that is a bound, both feet off between
        stances. crouch = 0, duty > 0.5 is the walk; so the walk and the run are one cycle
        with different parameters, and a course blends the parameters, not two foot paths
        (blending a duty-0.8 walk with a duty-0.33 run at one phase has both feet lifted
        with no thrust at half-blend, and the body fell at every change)."""
        if u < duty:
            w = u / duty
            return L / 2 - L * w, -crouch * (1.0 - ext * max(0.0, 2 * w - 1))
        w = (u - duty) / (1 - duty)
        return -L / 2 + L * w, -crouch + lift * math.sin(math.pi * w)

    def _walk_leg(self, u: float) -> tuple[float, float]:
        p = self.p
        return self._leg(u, p["step_len"], p["step_duty"], p["step_lift"], 0.0, 0.0)

    def _run_leg(self, u: float, ramp: float) -> tuple[float, float]:
        """The run's amplitudes (sweep, crouch, lift) scaled by ramp. The parked stance is
        nearly straight-legged (no room to extend), so the run lives run_crouch lower, and
        the whole sweep sits run_lean behind the hip so the toe-off extension pushes forward."""
        p = self.p
        x, z = self._leg(u, ramp * p["run_len"], p["run_duty"], ramp * p["run_lift"], ramp * p["run_crouch"], p["run_ext"])
        return x - ramp * p["run_lean"], z

    def _course_leg(self, u: float) -> tuple[float, float]:
        """The course's cycle: the walk's and the run's parameters mixed by the blend, the
        amplitudes (sweep, lift, crouch) scaled by the stepping amplitude."""
        p, a, b = self.p, self.amp, self.blend
        mix = lambda w, r: (1 - b) * p[w] + b * p[r]
        x, z = self._leg(u, a * mix("step_len", "run_len"), mix("step_duty", "run_duty"), a * mix("step_lift", "run_lift"),
                         a * b * p["run_crouch"], b * p["run_ext"])
        return x - a * b * p["run_lean"], z

    def _course_duty(self) -> float:
        return (1 - self.blend) * self.p["step_duty"] + self.blend * self.p["run_duty"]

    def course_gait(self, t: float) -> str:
        """The gait the course profile commands at time t."""
        acc = 0.0
        for g, d in self.profile:
            acc += d
            if t < acc:
                return g
        return self.profile[-1][0]

    def _course_update(self, t: float) -> None:
        """Advance the course state to t: the commanded gait sets the targets of the
        stepping amplitude (0 standing, 1 stepping) and of the walk/run mix, both slewed
        over trans_s (a stand is a walk of zero amplitude, so a stop after a run also
        brings the walk's balance genes and height feedback back); the shared phase runs
        at the blended cadence and is held at the left foot's swing start while stopped,
        so every start begins with a left step. A stop is a closing step: the gait keeps
        its amplitude until the left foot's next touchdown ("wait"); from there the phase
        rate decays linearly to zero over about trans_s (rounded to a whole number of cycles
        plus the half that ends with the right foot's landing; the stance sweep is the body's
        speed, so the body decelerates over its planted feet) and the right foot's last swing
        lands beside the left instead of ahead of it ("swing"); then the phase freezes with
        the feet together and the amplitude fades ("hold")."""
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        g = self.course_gait(t)
        amp_t = 0.0 if g == "stand" else 1.0
        blend_t = 1.0 if g == "run" else 0.0                           # a stand is a walk of zero amplitude
        step = dt / self.p["trans_s"]
        duty = self._course_duty()
        if amp_t > 0.0:
            self.closing = None
        elif self.amp > 0.0 and self.closing is None:
            self.closing = "wait"
        if self.closing in ("wait", "swing"):
            amp_t = self.amp                                           # keep stepping until the closing step has landed
        self.amp += max(-step, min(step, amp_t - self.amp))
        self.blend += max(-step, min(step, blend_t - self.blend))
        hz = (1 - self.blend) * self.p["step_hz"] + self.blend * self.p["run_hz"]
        if self.amp <= 0.0:
            self.cycles, self.closing = duty, None                     # parked at the left swing start
        elif self.closing == "swing":
            # braking: the rate hz * sqrt(1 - done / length) is a linear decay in time from hz at
            # the left touchdown to zero after `length` cycles, at phase 0.5, the right foot landed
            c0, length = self._brake
            self.cycles += hz * math.sqrt(max(0.0, 1.0 - (self.cycles - c0) / length)) * dt
            if self.cycles - c0 >= length - 1e-3:
                self.cycles, self.closing = c0 + length, "hold"
        elif self.closing != "hold":
            self.cycles += hz * dt
        prev, self.phase = self.phase, self.cycles % 1.0
        if self.closing == "wait" and self._crossed(prev, self.phase, 0.0):
            self.closing = "swing"                                     # the left foot has landed: the brake begins
            self.cycles, self.phase = math.floor(self.cycles + 0.5), 0.0
            self._brake = (self.cycles, max(0, round(hz * self.p["trans_s"] / 2 - 0.5)) + 0.5)

    @staticmethod
    def _crossed(prev: float, now: float, mark: float) -> bool:
        """Did the phase pass `mark` going from prev to now (mod 1)?"""
        return now != prev and ((mark - prev) % 1.0) <= ((now - prev) % 1.0)

    def gait_offsets(self, t: float) -> dict[str, tuple[float, float, bool]]:
        """Per side: (foot x offset, foot lift, height feedback on) for the legged
        objectives; zero for the others."""
        p = self.p
        if self.objective == "walk":                   # the cycle starts at the left foot's swing: the left steps first
            if self.contact_gait:
                return self._contact_walk(t)
            self.cycles = p["step_hz"] * t + p["step_duty"] + self.clock_off
            for s, ph in (("l", 0.0), ("r", 0.5)):
                u = (self.cycles + ph) % 1.0
                self.u_phase[s] = u
                self.swing_w[s] = 0.0 if u < p["step_duty"] else (u - p["step_duty"]) / (1.0 - p["step_duty"])
            return {s: (*self._walk_leg((self.cycles + ph) % 1.0), True) for s, ph in (("l", 0.0), ("r", 0.5))}
        if self.objective == "run":
            self.cycles = p["run_hz"] * t + p["run_duty"]
            ramp = min(1.0, t / RUN_RAMP_S)            # from the parked stance, not a 9 cm drop at t = 0
            return {s: (*self._run_leg((self.cycles + ph) % 1.0, ramp), False) for s, ph in (("l", 0.0), ("r", 0.5))}
        if self.objective == "course":
            out = {s: (*self._course_leg((self.phase + ph) % 1.0), True) for s, ph in (("l", 0.0), ("r", 0.5))}
            c0, length = self._brake
            if self.closing == "hold" or (self.closing == "swing" and self.cycles - c0 >= length - 0.5):
                # the closing step: the right foot's last swing lands beside the left
                duty = self._course_duty()
                u_r = (self.phase + 0.5) % 1.0
                w = 1.0 if self.closing == "hold" else max(0.0, min(1.0, (u_r - duty) / max(1e-6, 1.0 - duty)))
                xl, (xr, zr, fb) = out["l"][0], out["r"]
                out["r"] = ((1 - w) * xr + w * xl, zr, fb)
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

    def _contact_walk(self, t: float) -> dict[str, tuple[float, float, bool]]:
        """The walk cycle driven by the foot sensors (see the CATCH_S block of constants)."""
        p, sens = self.p, self.sensors
        hz, duty, L, lift = p["step_hz"], p["step_duty"], p["step_len"], p["step_lift"]
        dt = 0.0 if self._tw_prev is None else max(0.0, t - self._tw_prev)
        self._tw_prev = t
        phases = (("l", 0.0), ("r", 0.5))
        if not self._gait_started:
            # start: parked at the left swing start, shift ramping onto the right foot until it carries the body
            self.cycles = duty
            if self.instant_start or (sens.load["r"] >= START_LOAD_BW and t >= START_RAMP_S) or t >= START_TIMEOUT_S:
                self._gait_started = True
                self._set_anchor("r")                   # the right foot carries; the left swings from where it stands
                if self.anchor_measured:
                    self._swing_from["l"] = float(self.foot_meas["l"][0]) - self._x_bal["l"]
            return {s: (*self._walk_leg((self.cycles + ph) % 1.0), True) for s, ph in phases}
        rate = hz
        for s, ph in phases:
            u = (self.cycles + ph) % 1.0
            if u < duty:
                continue                                # stance: the sweep runs from the anchor at the clock's rate
            w = (u - duty) / (1.0 - duty)
            self._off[s] = self._off[s] + 1 if not sens.contact[s] else 0
            if self._off[s] >= LIFT_STEPS:
                self._lifted[s] = True
            self._loaded[s] = self._loaded[s] + 1 if (sens.contact[s] and sens.load[s] >= LAND_LOAD_BW) else 0
            if not self._landed[s] and self._lifted[s] and w >= SWING_MIN_FRAC and self._loaded[s] >= LAND_STEPS:
                self._landed[s] = True                  # early landing: pull the phase to this foot's touchdown
                self._set_anchor(s)
                self.contact_events["early"] += 1
                rate = max(rate, hz + (1.0 - u) / CATCH_S)
            elif not self._landed[s] and u + hz * dt >= 1.0 - 1e-9 and self._wait[s] < CONTACT_WAIT_S:
                self._wait[s] += dt                     # late: hold the phase, search down; the clock takes over after the wait
                self._search[s] = min(SEARCH_MAX_M, self._search[s] + SEARCH_DOWN_M_S * dt)
                rate = 0.0
                if self._wait[s] >= CONTACT_WAIT_S:
                    self.contact_events["late"] += 1
        prev = self.cycles
        self.cycles += rate * dt
        # the stance sweep is the body's speed: a phase pulled forward (early landing) or held (late)
        # must not jerk or stop the planted foot, so its anchor absorbs the difference from the clock
        for s, ph in phases:
            if (prev + ph) % 1.0 < duty and not (HOLD_STOPS_SWEEP and rate == 0.0):
                self._anchor[s] += L * (rate - hz) * dt / duty
        out = {}
        for s, ph in phases:
            u_prev, u = (prev + ph) % 1.0, (self.cycles + ph) % 1.0
            if self._crossed(u_prev, u, 0.0):           # touchdown mark: stance begins where the foot is
                if not self._landed[s]:
                    self._set_anchor(s)
                    self.contact_events["clock"] += 1
                self._landed[s], self._lifted[s], self._loaded[s], self._off[s] = False, False, 0, 0
                self._search[s], self._wait[s] = 0.0, 0.0
            if self._crossed(u_prev, u, duty):          # lift-off: the swing leaves from where the foot is
                self._swing_from[s] = (float(self.foot_meas[s][0]) - self._x_bal[s]) if self.anchor_measured else -L / 2
            if u < duty:
                gx, gz = self._anchor[s] - L * (u / duty), 0.0
                self.dy_hold[s], self.swing_w[s] = self.gait_dy[s], 0.0
            elif self._landed[s]:                       # down early: hold it there while the phase catches up
                gx, gz = self._anchor[s], 0.0
                self.dy_hold[s], self.swing_w[s] = self.gait_dy[s], 0.0
            else:
                w = (u - duty) / (1.0 - duty)
                x0 = self._swing_from[s]
                gx, gz = x0 + (L / 2 - x0) * w, lift * math.sin(math.pi * w) - self._search[s]
                self.dy_hold[s], self.swing_w[s] = self.gait_dy[s] * (1.0 - w), w
            out[s] = (gx, gz, True)
        return out

    def _catch_update(self, t: float, t_g: float, pitch: float, pitch_rate: float) -> None:
        """The catch step (task key catch_step, clock walk). On cobbles the walkers fell after a touchdown: the body
        pitched away at about 1 rad/s, the swing foot, held at its height below the tilted hip, rose with the body and
        never landed, and the stance foot tipped over its heel edge. So: when the pitch is past catch_pitch and still
        growing at catch_rate with a foot in the air, that foot is taken over (act places it level on the ground at the
        capture point) and the walk clock is frozen, so the stance foot stays down. When the catch foot carries
        LAND_LOAD_BW for LAND_STEPS steps the clock is re-phased with it at touchdown, and both feet's offsets from
        their new path fade out over half a stance. A catch that has not landed in catch_timeout lets the clock run."""
        p = self.p
        if self.catch_side is None:
            if (t >= CATCH_ARM_S and abs(pitch) > self.catch_pitch and pitch * pitch_rate > 0
                    and abs(pitch_rate) > self.catch_rate):
                # a foot in its swing that has actually left the floor (one still carrying load is not falling free)
                swing = [s for s in "lr" if self.swing_w[s] > 0.0 and self.sensors.load[s] < LAND_LOAD_BW]
                if swing:
                    self.catch_side, self.catch_t0, self.catch_land = swing[0], t, 0
                    self.catch_events += 1
            return
        s = self.catch_side
        self.clock_off -= p["step_hz"] * CONTROL_DT          # the clock holds still while the catch is in the air
        self.catch_land = self.catch_land + 1 if self.sensors.load[s] >= LAND_LOAD_BW else 0
        if self.catch_land >= LAND_STEPS and t - self.catch_t0 > 2 * CONTROL_DT:
            duty, hz = p["step_duty"], p["step_hz"]
            cyc = hz * t_g + duty + self.clock_off
            ph = {"l": 0.0, "r": 0.5}
            self.clock_off += (-(cyc + ph[s])) % 1.0          # the catch foot at touchdown
            cyc2 = hz * t_g + duty + self.clock_off
            fade = max(CONTROL_DT, 0.5 * duty / hz)
            for side in "lr":
                x_new = self._walk_leg((cyc2 + ph[side]) % 1.0)[0]
                self.x_bias[side] = self._gx_last[side] - x_new
                self.x_bias_rate[side] = abs(self.x_bias[side]) / fade
            self.catch_side = None
        elif t - self.catch_t0 > self.catch_timeout:
            self.catch_side = None

    def _set_anchor(self, s: str) -> None:
        """Stance from the measured foot: the sweep starts at the foot's x relative to the balance
        target, and the lateral target keeps the landing error so the planted foot is not asked
        to slide to the nominal track."""
        if not self.anchor_measured:
            self._anchor[s], self.gait_dy[s] = self.p["step_len"] / 2, 0.0
            return
        self._anchor[s] = float(self.foot_meas[s][0]) - self._x_bal[s]
        self.gait_dy[s] = float(self.foot_meas[s][1]) - self._dy_tgt[s]

    def body_shift(self, t: float) -> float:
        """Walk: lateral body offset (+ = left) over the feet, furthest toward the
        stance side mid-swing of the other foot; zero for the other objectives."""
        if self.objective == "walk" and self.contact_gait:
            u, scale = self.cycles % 1.0, (1.0 if self.instant_start else min(1.0, t / START_RAMP_S))
        elif self.objective == "walk":
            u, scale = (self.p["step_hz"] * t + self.p["step_duty"]) % 1.0, 1.0
        elif self.objective == "run":
            u, scale = (self.p["run_hz"] * t + self.p["run_duty"]) % 1.0, 1.0
        elif self.objective == "course":
            u, scale = self.phase, min(1.0, self.amp / 0.2)              # the sway is there as soon as the feet move
        else:
            return 0.0
        # 1.0 = hip over the stance foot: the hip point, or the foot where inboard feet have put it
        amp = scale * self.p["walk_shift"] * max(0.0, self.p["hip_width"] / 2 + min(0.0, self.stance_w))
        return amp * math.cos(2 * math.pi * (u - self.p["shift_phase"]))

    def _feet_down(self, data: mujoco.MjData) -> bool:
        for ci in range(data.ncon):
            g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
            if (g1 == self.floor_geom and g2 in self.foot_geoms) or (g2 == self.floor_geom and g1 in self.foot_geoms):
                return True
        return False

    def _level_soles(self, data: mujoco.MjData) -> None:
        """A passive parallelogram from the hip keeps the sole parallel to the body
        whatever the knee does, and the ankle spring acts about that. Without it the
        spring rest is the parked shin angle: bend the knee and the spring (25 N m/rad,
        stiff enough to stand on) tilts the sole with the shin, the foot rides on one
        edge, and the body is on a knife edge. Positive ankle = toe down, so the sole
        is level in the body frame at -(thigh + knee), toe up by ankle_rest."""
        if not LEVEL_SOLES:
            return
        for s in ("l", "r"):
            shin = sum(float(data.qpos[a]) for a in self.chain_q[s])     # the link the foot hangs from
            self.model.qpos_spring[self.ankle_q[s]] = -shin - self.p["ankle_rest"]

    def _slew(self, name: str, target: float) -> float:
        step = (self.roll_servo.max_rad_s / self.roll_ratio if name.startswith("hip_roll") else self.servo.max_rad_s) * CONTROL_DT
        if self.real_servo and self._data is not None:
            # speed-torque line: the servo slews at (1 - load / stall) of its no-load speed
            a = self.idx[name]
            frac = abs(float(self._data.actuator_force[a])) / max(1e-9, float(self.model.actuator_forcerange[a][1]))
            step *= max(SERVO_MIN_SPEED_FRAC, 1.0 - min(1.0, frac))
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
        self._data = data
        self.sensors.update(data)
        if self.measure_feet:
            R = data.xmat[self.hip_id].reshape(3, 3)
            for s in "lr":
                self.foot_meas[s] = R.T @ (data.xpos[self.foot_body[s]] - data.xpos[self.hip_id]) - self.hip_point[s]
        pos = data.qpos[:3].copy()
        self._level_soles(data)
        if self.objective == "course":
            self._course_update(t)
        # how much of a run this is: the run's balance genes and fan rules blend in with it
        b_run = self.blend if self.objective == "course" else (1.0 if self.objective == "run" else 0.0)
        runish = b_run > 0.5
        if self.p0 is None:
            self.p0 = pos.copy()
        quat = data.qpos[3:7]
        roll, pitch = quat_to_roll_pitch(quat)
        yaw = quat_to_yaw(quat)
        v_world = data.qvel[:3]
        roll_rate, pitch_rate, yaw_rate = float(data.qvel[3]), float(data.qvel[4]), float(data.qvel[5])
        cy, sy = math.cos(yaw), math.sin(yaw)
        # latency compensation: the tilt the body will have when this command lands, and the gait
        # time it lands at (self.lead = 0 leaves both as measured)
        roll, pitch = roll + self.lead * roll_rate, pitch + self.lead * pitch_rate
        t_g = t + self.lead

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
        # legged: airborne means no foot on the floor for FLIGHT_PUSH_S (a fast walk has a
        # flight gap at every step, and the hip-height test flickered on every bob; either
        # sent the fans into a forward push at idle throttle, which is a 90 deg pitch command
        # and banged the pitch servos between their stops at each footfall)
        if self.legged:
            if self._feet_down(data):
                self.air_since = None
            elif self.air_since is None:
                self.air_since = t
            gate = 0.0 if runish else FLIGHT_PUSH_S     # a run's flight is the push phase
            airborne = self.air_since is not None and t - self.air_since >= gate
        else:
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
        # a run with run_fan ~ 0 asks nothing of the fans in flight either (a push at idle
        # throttle is only a pitch command banging between its stops)
        fan_push = airborne and (not runish or p["run_fan"] > 0.02)
        push = 0.0 if (self.legged and not fan_push) else 1.0
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
            # stance_x puts the sole's support under the centre of mass rather than the ankle
            # under the hip: the CoM sits ahead of the hip and the sole's support is only
            # +-6 cm about the ankle, so the parked stance stands on its front edge (a live
            # CoM feed-forward was tried and oscillates: the legs are a quarter of the mass)
            # the run has its own offset, pitch and speed gains (run_*), mixed in with b_run
            sx = (1 - b_run) * p["stance_x"] + b_run * p["run_stance_x"]
            kp = (1 - b_run) * p["kp_foot_leg"] + b_run * p["run_kp_leg"]
            kv = (1 - b_run) * p["kv_foot_x"] + b_run * p["run_kv_x"]
            if self.capture:
                kv = 0.0                                 # the velocity term is the swing foot's capture point below
            x_f = self.x_nom + sx + kp * pitch + p["kd_foot_leg"] * pitch_rate + self.x_int + kv * float(v_b[0])
            if self.cop_gain and self.objective in ("walk", "stand"):
                # heel-toe feedback (task keys cop_gain m, cop_dead 0-1): the planted feet's load-weighted centre of
                # pressure from the heel and toe pads, as a fraction of the pad spacing (-1 heel .. +1 toe); past the
                # deadband the feet shift the same way (heel -> feet back, the hip comes forward over them). In every
                # traced cobble and shove fall the pressure pinned at an edge a fraction of a second before the tip
                # could not be undone.
                sens, num, den = self.sensors, 0.0, 0.0
                for s in "lr":
                    if sens.contact[s] and sens.load[s] > 0.0:
                        num += sens.load[s] * sens.cop[s]
                        den += sens.load[s]
                if den > 0.0:
                    c_n = max(-1.0, min(1.0, num / den / max(1e-6, sens.pad_x)))
                    e = math.copysign(max(0.0, abs(c_n) - self.cop_dead) / max(1e-6, 1.0 - self.cop_dead), c_n)
                    x_f += self.cop_gain * e
                    self.cop_e = e
        else:
            x_f = (self.x_nom - d_feet
                   + p["kp_foot_x"] * (pitch - pitch_ref) + p["kd_foot_x"] * pitch_rate)
        dh_roll = p["kp_roll_lvl"] * roll + p["kd_roll_lvl"] * roll_rate   # left side up -> shorten left
        # feet toward the fall and toward the lateral motion: with kp_roll_y ~ leg drop and
        # kv_foot_y ~ sqrt(drop / g) the swing foot lands on the capture point
        dy_f = (-self.body_shift(t_g) - p["kp_roll_y"] * roll - p["kd_roll_y"] * roll_rate
                + (0.0 if self.capture else p["kv_foot_y"]) * float(v_b[1]))
        # the swing foot's landing at the capture point: hip velocity x the pendulum time, in x and y
        T_c = math.sqrt(max(0.05, h_com) / G)
        cap_x, cap_y = (float(v_b[0]) * T_c, float(v_b[1]) * T_c) if self.capture else (0.0, 0.0)
        if self.catch_on:
            self._catch_update(t, t_g, pitch, pitch_rate)
            if not hasattr(self, "_gx_last"):
                self._gx_last = {"l": 0.0, "r": 0.0}
            R_w = data.xmat[self.hip_id].reshape(3, 3)
            R_h = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ R_w     # hip frame -> heading frame
            # the catch crouch: down at CROUCH_RATE while a catch is in the air, back up over half a stance after
            rate = CROUCH_RATE if self.catch_side else self.catch_crouch / max(CONTROL_DT, 0.5 * p["step_duty"] / p["step_hz"])
            target = self.catch_crouch if self.catch_side else 0.0
            self.crouch_now += max(-rate * CONTROL_DT, min(rate * CONTROL_DT, target - self.crouch_now))
        if (self.level_legs or self.level_swing) and self.legged:
            # the body's pitch about the heading frame's y axis, from its x axis: Ry(th) takes x to (cos th, 0, -sin th)
            R_hl = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ data.xmat[self.hip_id].reshape(3, 3)
            th_body = math.atan2(-float(R_hl[2, 0]), float(R_hl[0, 0]))
        gait = self.gait_offsets(t_g)
        hfb_scale = 1.0 - self.blend if self.objective == "course" else 1.0   # the run sets its own height
        stance_w = self.stance_w
        for s, sign in (("l", +1.0), ("r", -1.0)):
            gx, gz, fb = gait[s]
            h = max(0.02, self.h_nom + (h_fb * hfb_scale if fb else 0.0) + gz - sign * dh_roll)
            dy = dy_f + sign * stance_w                           # + = foot to the left, both legs
            self._x_bal[s], self._dy_tgt[s] = x_f, dy
            if self.contact_gait:
                dy += self.dy_hold[s]
            sw = self.swing_w[s]
            gx, dy = gx + sw * cap_x, dy + sw * cap_y
            if self.catch_on:
                if s == self.catch_side:
                    # the catch foot, placed in the level heading frame and turned back into the tilted hip frame:
                    # fore-aft at the capture point of the falling body (centre of mass + hip speed x pendulum time),
                    # down at the stance foot's ground less catch_down, sideways where the gait had it
                    o = "r" if s == "l" else "l"
                    hp_h = R_h @ self.hip_point[s]
                    lat_h = float((R_h @ np.array([x_f + gx, dy, -h]))[1])
                    com_x = float((R_h @ np.array([0.0, 0.0, COM_ABOVE_HIP]))[0])
                    x_h = com_x + self.catch_gain * float(v_b[0]) * T_c - float(hp_h[0])
                    z_h = (float(data.xpos[self.foot_body[o]][2]) - self.catch_down) - (float(data.xpos[self.hip_id][2]) + float(hp_h[2]))
                    d_b = R_h.T @ np.array([x_h, lat_h, z_h])
                    gx, dy, h = float(d_b[0]) - x_f, float(d_b[1]), max(0.02, -float(d_b[2]))
                else:
                    h = max(0.02, h - self.crouch_now)
                    b = self.x_bias[s]
                    if b != 0.0:
                        step = self.x_bias_rate[s] * CONTROL_DT
                        self.x_bias[s] = 0.0 if abs(b) <= step else b - math.copysign(step, b)
                    gx += self.x_bias[s]
                self._gx_last[s] = gx
            if (self.level_legs or self.level_swing) and self.legged:
                # the target (x_f + gx, dy, -h) read in the pitch-levelled frame, turned into the hip frame: Ry(th)^T
                k = self.level_legs
                if self.level_swing and self.objective == "walk":
                    u, duty = self.u_phase[s], p["step_duty"]
                    w_lv = 1.0 if u >= duty else max(0.0, 1.0 - (u / duty) / LEVEL_FADE)
                    k = max(k, self.level_swing * w_lv)
                th = k * th_body
                c_th, s_th = math.cos(th), math.sin(th)
                xv, zv = x_f + gx, -h
                xb, zb = c_th * xv - s_th * zv, s_th * xv + c_th * zv
                gx, h = xb - x_f, max(0.02, -zb)
            # the roll that puts the ankle dy to the left and h below the leg's hip point: with
            # the pivot roll_arm inboard of that point (pantograph body) the leg point itself
            # swings, so the roll is the target's angle about the pivot less the leg's own, and
            # the leg's length in its plane follows from the pivot distance
            L = self.roll_arm
            if L == 0.0:
                roll_cmd, drop = math.atan2(dy, h), math.hypot(h, dy)
            else:
                yt = sign * L + dy
                drop = math.sqrt(max(1e-4, yt * yt + h * h - L * L))
                roll_cmd = math.atan2(yt, h) - math.atan2(sign * L, drop)
            roll_cmd = max(-self.roll_range, min(self.roll_range, roll_cmd))
            c[i[f"hip_roll_{s}"]] = self._slew(f"hip_roll_{s}", roll_cmd)
            self.foot_target[s] = self.hip_point[s] + np.array([x_f + gx, dy, -h])
            if self.panto:
                # both horns from the ankle target; an unreachable target gives the nearest pose
                # clear of the folds, a failed solve holds the last one
                xa, za = x_f + gx, -drop
                if self.reach_clamp:
                    r_a = math.hypot(xa, za)
                    if r_a > self.reach_max:                 # radially, toward the hip point
                        xa, za = xa * self.reach_max / r_a, za * self.reach_max / r_a
                hf, hr, res = panto_ik(p, xa, za, self.ik_guess[s])
                if self.reach_clamp and not res < 1e-4:      # off the branch: solve again from the crouch
                    hf2, hr2, res2 = panto_ik(p, xa, za, self.ik_nominal)
                    if res2 < res:
                        hf, hr, res = hf2, hr2, res2
                if math.isfinite(res) and panto_pose_allowed(p, hf, hr):
                    self.ik_guess[s] = (hf, hr)
                hf, hr = self.ik_guess[s]
                c[i[f"hip_pitch_{s}"]] = self._slew(f"hip_pitch_{s}", hf)
                c[i[f"knee_{s}"]] = self.horn_cmd[s] = self._slew(f"knee_{s}", hr)
                continue
            thigh, shin = angles_from_foot(p, x_f + gx, -drop)
            thigh_cmd, horn_cmd = self._drive_commands(s, thigh, shin)
            self.horn_cmd[s] = horn_cmd                      # the shin servo's own angle, for traces
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
            lift = (1 - b_run) * p["step_lift"] + b_run * p["run_lift"] if self.objective in ("walk", "run", "course") else 1.0
            assist_l, assist_r = (p["step_fan"] * max(0.0, g[1]) / lift for g in (gait["l"], gait["r"]))
            # run: the fans spin up to run_fan while no foot is down and push the body forward
            base = max(p["fan_base"], b_run * p["run_fan"]) if (runish and airborne) else p["fan_base"]
            thr_l = max(0.0, min(1.0, base - diff + assist_l))
            thr_r = max(0.0, min(1.0, base + diff + assist_r))
            t_vert = max(0.05, base) * self.t_max
            # pitching a fan that makes no thrust does nothing but bang the pitch servo between
            # its stops on every footfall, so the attitude terms fade out with the throttle
            auth = min(1.0, 0.5 * (thr_l + thr_r) / FAN_AUTH_THR)
            common = (math.atan2(f_fan[0] / 2, t_vert) + p["fan_pitch_bias"]
                      - auth * (p["fan_kp_pitch"] * (pitch - pitch_ref) + p["fan_kd_pitch"] * pitch_rate))
            d_yaw = auth * p["fan_kp_yaw"] * (omega_des - yaw_rate)
            fan_l = max(-FAN_PITCH_RANGE, min(FAN_PITCH_RANGE, common - d_yaw))
            fan_r = max(-FAN_PITCH_RANGE, min(FAN_PITCH_RANGE, common + d_yaw))
        else:
            thr_l = thr_r = fan_l = fan_r = 0.0
        self.dbg = {"pitch": pitch, "pitch_ref": pitch_ref, "a_des": float(a_des[0]), "a_tread": a_tread,
                    "d_feet": d_feet, "v_bal": v_bal, "x_f": x_f, "h": self.h_nom + h_fb, "dy_f": dy_f,
                    "f_fan_x": float(f_fan[0]), "amp": self.amp, "blend": self.blend, "phase": self.phase,
                    "fan_l": fan_l, "fan_r": fan_r, "thr": (thr_l, thr_r)}
        c[i["fan_thrust_l"]], c[i["fan_thrust_r"]] = thr_l, thr_r
        c[i["fan_pitch_l"]], c[i["fan_pitch_r"]] = fan_l, fan_r
        self.last_throttle = (thr_l, thr_r)
        if self.real_servo:
            c = self._servo_link(c)
        return c

    def _servo_link(self, c: np.ndarray) -> np.ndarray:
        """What the six leg servos actually receive: the command issued SERVO_LATENCY_S ago,
        refreshed at SERVO_CMD_HZ and held between refreshes, and moved only when the change
        clears the pulse deadband. The fans and treads pass straight through."""
        c = c.copy()
        issued = c[self.servo_acts].copy()
        self._servo_queue.append(issued)
        delay = max(1, int(round(SERVO_LATENCY_S / CONTROL_DT)))
        if len(self._servo_queue) > delay + 1:
            self._servo_queue.pop(0)
        arrived = self._servo_queue[0]
        hold = max(1, int(round(1.0 / (SERVO_CMD_HZ * CONTROL_DT))))
        n = len(self._servo_queue)
        if self._servo_held is None:
            self._servo_held = arrived.copy()
        elif (n + self._tick) % hold == 0:
            move = np.abs(arrived - self._servo_held) >= SERVO_CMD_DEADBAND_RAD
            self._servo_held = np.where(move, arrived, self._servo_held)
        self._tick += 1
        c[self.servo_acts] = self._servo_held
        return c
