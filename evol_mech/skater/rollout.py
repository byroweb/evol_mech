"""Single-episode simulation of one skater genome on one task.

Task dict: {"name", "terrain", "objective": "skate" | "crab" | "slalom" | "stand"
| "walk" | "run" | "hop" | "broad_jump", "duration", "weight", "speed" (m/s, skate/crab/
slalom/walk/broad_jump), "amplitude" and "wavelength" (slalom path
y = A sin(2 pi x / lambda))}.

Fitness (stage-1 tasks from the design doc):
  skate   mean speed along +x, scaled by the fraction of time upright,
          minus ride-quality cost (RMS hip acceleration) and lateral drift
  crab    the same along +y, drift along x penalised
  slalom  mean speed along +x minus lateral path error, foot scrub and
          cones hit; cones stand on the centreline at the path's lateral
          extremes (x = lambda/4 + n lambda/2), so the skater rounds each one
          on alternate sides
  stand   time upright minus drift, shoved every PERTURB_PERIOD seconds
  walk    progress of the feet (mean foot x) per second of the task, plus
          `stride_weight` (1.0) x the realized stride (progress per step,
          capped at STRIDE_CAP_FRAC of the leg length so the splits do not
          pay: larger, realistic steps for the same speed; a task may also
          cap the cadence with `step_hz_max`), minus a servo
          chatter cost (`chatter_cost`, CHATTER_COST per objective, x the
          RMS angular acceleration of the eight servo commands: hip roll,
          thigh, shin horn and fan pitch on each side; the fans count so a
          pitch gain that bangs an idle fan between its stops is paid for)
          and `sat_cost` (SAT_COST) x the fraction of control steps with a
          leg servo moving at its rate limit (motors slammed to their stops),
          minus lateral drift (no ride-quality cost by default: step impacts are
          not a fault, and at skate's 0.02 they outweighed the progress of a
          0.1 m/s walk); treads locked (the gait cycle moves the feet). The
          fall penalty is small (0.1) and scales with the time left: the
          episode ends at a fall, so what counts is the distance walked
          before it, and a gait that steps and falls late outscores one that
          stands still (the flat -0.5 made standing the attractor), while a
          fall in the first second pays it all (with no penalty, falling at
          once avoided every running cost and scored zero). The feet, not
          the hip, measure the progress so a forward dive earns nothing.
  run     a bounding gait (controller.gait_offsets: stance sweep run_len at
          controller.gait_offsets: stance sweep run_len at run_hz with run_duty
          in stance, the hip run_crouch low and run_ext of that recovered by
          toe-off, swing foot retracted run_lift; the fans spin to run_fan
          while no foot is down and push forward) + `stride_weight` (0.5) x the realized stride
          (capped at RUN_STRIDE_CAP_FRAC of the leg) + `flight_weight` (0.3)
          x the share of the upright time in real flights (both feet off for
          FLIGHT_MIN_S with the hip rising FLIGHT_MIN_RISE; a fast walk loses
          foot contact a fifth of the time without the hip rising, and that
          flicker is not a bound), minus the walk's chatter, rate-limit and
          scaled fall costs
  hop     sum of the three highest ballistic rises of the hip (apex above its
          take-off height) minus drift, so repeatable high hops score and a
          chatter of tiny bounces does not; treads locked
  broad_jump  longest flight (hip x from take-off to landing) plus a little
          of the total distance; treads locked
A flight is both feet off the floor for at least FLIGHT_MIN_S with the hip
rising at least FLIGHT_MIN_RISE above its take-off height (sole bounce and
the spawn drop are not flights).
All tasks: a fall costs `fall_cost` (task key; default FALL_COST per
objective: 0.5, walk 0.1 x the fraction of the task left), plus fan energy,
leg + tread motor energy, and the
distal-mass penalty (each leg above LEG_MASS_FRACTION_TARGET of total mass).
A fall is a tipped or dropped hip, or anything but a foot touching the floor.
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from ..sim import terrain as terrain_mod
from ..sim.controller import quat_to_roll_pitch
from ..sim.rollout import (CONTROL_DT, SUBSTEPS, INVALID_FITNESS, UNSTABLE_FITNESS,
                           PERTURB_PERIOD, PERTURB_LEN, PERTURB_ACCEL, TERRAIN_ZMAX,
                           _body_touching_floor, _invalid_result)
from .spec import decode
from .mjcf import build_mjcf, nominal_qpos, ACTUATOR_NAMES
from .structure import InvalidBody, nominal_height, leg_mass_fraction, leg_reach, LEG_MASS_FRACTION_TARGET
from .controller import SkaterController, COURSE_PROFILE

OBJECTIVES = ("skate", "crab", "slalom", "stand", "walk", "run", "course", "hop", "broad_jump")
GAIT_HZ = {"walk": "step_hz", "run": "run_hz"}   # the cycle-rate gene of each stepping objective
STEPPING = ("walk", "run", "course")             # objectives whose fall cost scales with the time left
# course scoring (see _course_scores): per-segment terms with the walk's and run's weights,
# a stop cost on the stands, and a bonus per gait change survived with little pitch excursion
TRANS_WIN_S = 2.0               # the window after a gait change scored as the transition (and left out of the segment's steady window)
COURSE_SPEED_WEIGHT = {"walk": 0.3, "run": 2.0}   # the walk's as WALK_SPEED_WEIGHT: a walk is judged on grace
COURSE_STRIDE_WEIGHT = {"walk": 3.0, "run": 0.5}  # the walk's as WALK_STRIDE_WEIGHT
COURSE_FLIGHT_WEIGHT = 0.0      # as RUN_FLIGHT_WEIGHT: the run segment is gated by its flight share, not paid for it
STOP_COST = 1.0                 # fitness per m the feet move during a stand (after the change)
SETTLE_COST = 0.5               # fitness per rad of pitch RMS during a stand
TRANS_WEIGHT = 0.3              # bonus per transition, scaled down by the pitch excursion over TRANS_PITCH
TRANS_PITCH = 0.5               # rad of peak pitch in the window at which the transition bonus is nil
T90_SMOOTH_S = 0.5              # the start metric: time to 90 % of the segment's steady speed, speed smoothed over this
COST_REF_S = 8.0                # course: energy and drift costs are scaled to this task length (its scores are per second;
                                # a 36 s course would otherwise pay 4.5 x the fan energy the 8 s walk and run tasks evolved under)
STRIDE_CAP_FRAC = 0.75          # realized stride rewarded up to this fraction of thigh + shin (human step/leg ~0.7-0.8)
RUN_STRIDE_CAP_FRAC = 1.2       # run: step per leg length of a human jog ~1.0-1.5
FLIGHT_FRAC_CAP = 0.5           # run: share of the upright time in real flights rewarded up to this (a bound, not a single leap)
# run scoring: a run is for speed. Flight is the gate, not the prize: the speed, stride and flight-length
# terms are scaled by min(1, flight share / FLIGHT_GATE), so a fast walk with no flight earns nothing
# and a hop earns nothing more for hanging in the air; the flight's length is rewarded (per m, capped)
# and the hip's bob is charged. The round-2 bound hopped: 22 flights in 8 s, 6 cm long at a 2 cm apex,
# half the time airborne going nowhere, 0.28 m/s against the walk's 0.4, because the flight share paid
# 0.15 of fitness whatever the flight covered.
FLIGHT_GATE = 0.15              # run: share of the upright time in real flights for full credit
FLIGHT_LEN_WEIGHT = 1.0         # run: fitness per m of mean flight length ...
FLIGHT_LEN_CAP = 0.30           # ... up to this (a leg length)
RUN_SPEED_WEIGHT = 2.0          # run: fitness per m/s of the feet (was 1.0)
RUN_FLIGHT_WEIGHT = 0.0         # run: fitness per unit of flight share (was 0.3; the gate replaces it)
RUN_BOB_COST = 2.0              # run: fitness per m of hip height standard deviation
CHATTER_COST = {"walk": 1e-4, "run": 1e-4, "course": 1e-4}   # fitness per rad/s^2 of servo command acceleration RMS (0 elsewhere)
SAT_COST = {"walk": 0.0, "run": 0.1, "course": 0.1}      # fitness x the fraction of control steps with a leg servo at its rate limit
                                # (walk 0: the slamming is in its grace factor instead)
FALL_COST = {"walk": 0.3, "run": 0.1, "course": 0.1}     # default fall penalty per objective (0.5 otherwise); on the walk it
                                # scales with the time left, so an early fall pays it all and a late one little
                                # (walk 0.3: under the grace score a 12 s walk is worth ~0.15, and at 0.1 a gait that
                                # fell at 1 s, paying no chatter, fan or sway, won a round)
ACCEL_COST = {"walk": 0.0, "run": 0.0, "course": 0.0}    # default ride-quality cost per objective (0.02 otherwise): a walk's
                                # step impacts are not a ride-quality fault, and at 0.02 they outweighed
                                # the progress of a 0.1 m/s walk, so standing still won
FLIGHT_MIN_S = 0.04             # both feet off the floor at least this long ...
FLIGHT_MIN_RISE = 0.01          # ... with the hip rising at least this (m) counts as a flight
CLEAR_WEIGHT = {"walk": 4.0, "run": 2.0, "course": 2.0}   # fitness per m of swing-foot clearance (median over swings), capped at CLEAR_CAP
                                # (walk 4, outside its grace factor: scaled by grace at 2 the first grace round shuffled at 8 mm)
COURSE_CLEAR_WEIGHT = {"walk": 4.0, "run": 2.0}           # the course's per gait, as the plain objectives
CLEAR_CAP = 0.05                # a shuffle trips on a pebble; 5 cm of clearance is paid for, more is not
# walk quality: a walk is judged on grace, not speed. Its gait score (the stride first, a little
# speed, the clearance) is multiplied by a grace factor
#     1 / (1 + sway/SWAY_REF + bob/BOB_REF + surge/SURGE_REF + sat/SAT_REF)
# so a smoother body is worth more of the same stride while standing still is still worth nothing
# (a subtractive smoothness cost had made the stander win), and the fans' mean throttle is
# charged on top of their energy (FAN_DUTY_COST), so a walk that leans on its fans pays twice.
# The clearance bonus stays outside the factor: picking the feet up costs some sway, and with
# the bonus scaled by grace the first grace round traded the lift away (3.4 cm -> 8 mm).
# The round-6 walker: sway 0.11 rad (roll 4.6 deg RMS), bob 3 mm, surge 0.06 m/s, a leg servo at
# its slew limit 89 % of the time, grace 0.39, fan duty 6 %.
WALK_SPEED_WEIGHT = 0.3         # walk: fitness per m/s of the feet (was 1.0; speed is not what a walk is for)
WALK_STRIDE_WEIGHT = 3.0        # walk: fitness per m of realized stride (was 1.0; the stride is what a walk is judged on)
WALK_SPEED_GATE = 0.15          # walk: the gait score is scaled by min(1, speed / this), so a walk is still a walk: the first
STAND_COST = 0.3                # walk: a stander pays this x (1 - speed gate): standing still in a walk task is a fall at t = 0 (the pantograph's first round was won by one)
# walk: task key `stride_min` (m, default 0 = off) is a stride floor: the speed gate is also min(1, stride / stride_min),
# so a shuffle keeps that fraction of its gait score and clearance bonus and pays the rest of the stand cost (the
# real-servo round of phase 1 chose 2.7 cm steps that stood for 12 s over any gait that travelled)
# walk: task key `clear_min` (m, default 0 = off) is a clearance floor on the same gate: min(1, swing clearance / clear_min)
# (round 1f, the capture-point round on the strider, won with 1.0 cm of clearance where its seed had 3.0: dragging the feet
# was cheaper than lifting them and nothing in the score said otherwise)
                                # course round under the grace score shrank its walk segments to a 3 cm shuffle at 0.03 m/s,
                                # safe and graceful and going nowhere (a slow walk is fine, standing is not)
SWAY_REF = 0.20                 # rad of body pitch + roll RMS that halves the gait score on its own
BOB_REF = 0.02                  # m of hip height standard deviation, likewise
SURGE_REF = 0.15                # m/s of forward hip speed standard deviation, likewise
WEAVE_REF = 0.02                # m of lateral hip excursion (standard deviation about a WEAVE_WIN_S moving average, so
                                # a slow drift is not a weave), likewise (the swagger: the first grace round walked a
                                # 5 cm track with a 0.75 weight shift, a cowboy)
WEAVE_WIN_S = 1.5
TRACK_REF = 0.05                # m of track (the feet's lateral separation) beyond the hip width, likewise: the wide stance
                                # is the other half of the cowboy; the grace rounds walked a 24 cm track on a 14 cm hip
SAT_REF = 2.0                   # fraction of the time a leg servo is at its slew limit, likewise (1 = always: a quarter off)
FAN_DUTY_COST = {"walk": 1.0, "course": 1.0}   # fitness x the fans' mean throttle (0-1) on a walk (course: per walk segment)
HOP_TOP_N = 3                   # hop fitness sums the highest apex rises of this many flights
LEG_MASS_PENALTY = 5.0          # fitness per unit of leg fraction above target, per leg
CONE_HEIGHT = 0.30              # a small traffic cone: 30 cm tall, 10 cm base radius
CONE_BASE_R = 0.10
CONE_LAYERS = 5                 # stacked cylinders (MuJoCo has no cone primitive)
CONE_COST = 0.1                 # default fitness per cone touched (task key "cone_cost")


def slalom_cones(task: dict) -> list[float]:
    """x positions of the slalom cones: on the centreline at the path's lateral
    extremes, one every half wavelength, over the distance the task can cover."""
    lam = float(task.get("wavelength", 3.0))
    course = float(task.get("speed", 1.0)) * float(task.get("duration", 0.0)) + lam
    return [lam / 4 + n * lam / 2 for n in range(int(course / (lam / 2)) + 1)
            if lam / 4 + n * lam / 2 < course]


def _cones_xml(xs: list[float], hdata, zmax: float) -> str:
    """Static orange cones with a white band, tapered as stacked cylinders."""
    out = []
    h = CONE_HEIGHT / CONE_LAYERS
    for n, x in enumerate(xs):
        z0 = terrain_mod.height_at(hdata, zmax, x, 0.0)
        for k in range(CONE_LAYERS):
            r = CONE_BASE_R * (1 - 0.75 * (k + 0.5) / CONE_LAYERS)
            rgba = "0.95 0.95 0.95 1" if k == CONE_LAYERS - 2 else "1.0 0.45 0.05 1"
            out.append(f'<geom name="cone_{n}_{k}" type="cylinder" size="{r:.4f} {h / 2:.4f}" '
                       f'pos="{x:.4f} 0 {z0 + (k + 0.5) * h:.4f}" rgba="{rgba}"/>')
    return "\n    ".join(out)


LIDAR_KEYS = ("lidar", "lidar_pitch_deg", "lidar_z")
WAIST_KEYS = ("waist_stiffness", "waist_damping", "waist_pivot", "waist_link")


def make_model(genome: np.ndarray, terrain_kind: str, rng: np.random.Generator, task: dict | None = None):
    """Compile a model. Raises InvalidBody for un-buildable genomes. With a slalom
    task the cones are placed in the scene."""
    p = decode(genome)
    for hz in GAIT_HZ.values():                   # a task may cap the cadence (slower, longer steps)
        if task and f"{hz}_max" in task:
            p[hz] = min(p[hz], float(task[f"{hz}_max"]))
    for k in LIDAR_KEYS:                          # the core's VL53L8CX (mjcf._lidar, lidar.py); off by default
        if task and task.get(k) is not None:
            p[k] = float(task[k])
    for k in WAIST_KEYS:                          # the waist roll joint is a task option (mjcf._waist); off by default
        if task and task.get(k) is not None:
            p[k] = float(task[k])
    slope = {k: float(task[k]) for k in ("slope_deg", "slope_heading_deg") if task and k in task}
    asset, geom, hdata = terrain_mod.build(terrain_kind, rng, **slope)
    zmax = TERRAIN_ZMAX.get(terrain_kind, 0.0)
    scene = _cones_xml(slalom_cones(task), hdata, zmax) if task and task.get("objective") == "slalom" else ""
    xml = build_mjcf(p, asset, geom, zmax, scene)
    model = mujoco.MjModel.from_xml_string(xml)
    if hdata is not None:
        model.hfield_data[:] = hdata.ravel()
    return model, p


def make_data(model: mujoco.MjModel, p: dict[str, float]) -> mujoco.MjData:
    """MjData in the parked crouch with the actuators already targeting it."""
    data = mujoco.MjData(model)
    data.qpos[:] = nominal_qpos(model, p)
    mujoco.mj_forward(model, data)
    return data


def _fallen(z: float, roll: float, pitch: float, h_nom: float, pitch_max: float = 1.0) -> bool:
    """pitch_max (rad): task key fall_pitch_deg sets it (the user: a core pitched 20 deg is an imminent fall, and no
    traced walker came back from past ~15 deg, forward or back); 1 rad by default, every old score."""
    return z < 0.45 * h_nom or abs(roll) > 1.0 or abs(pitch) > pitch_max


def rollout(genome: np.ndarray, task: dict, seed: int = 0, *,
            fans_enabled: bool = True, fan_energy_cost: float = 0.2,
            motor_energy_cost: float = 0.002, leg_mass_penalty: float | None = None,
            frame_cb=None, lidar_cb=None, stop_on_fall: bool = True, **_ignored) -> dict:
    rng = np.random.default_rng(seed)
    objective = task["objective"]
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")
    if objective == "course":                        # the gait schedule sets the duration
        profile = [(str(g), float(d)) for g, d in task.get("profile", COURSE_PROFILE)]
        task = {**task, "profile": profile, "duration": float(task.get("duration") or sum(d for _, d in profile))}
    try:
        model, p = make_model(genome, task["terrain"], rng, task)
    except InvalidBody as e:
        return _invalid_result(str(e))
    data = make_data(model, p)
    if task.get("pose_noise_deg"):
        # a random initial tilt (roll and pitch, uniform within +-pose_noise_deg): a gait that only
        # works from one exact start is on a knife edge (a 1e-16 genome change flipped the first
        # course best from a 36 s completion into a fall at 16 s), and this is what widens it
        rx, ry = np.radians(rng.uniform(-1, 1, 2) * float(task["pose_noise_deg"]))
        q = np.array([1.0, rx / 2, ry / 2, 0.0])
        data.qpos[3:7] = q / np.linalg.norm(q)
        data.qpos[2] += 0.005
        mujoco.mj_forward(model, data)
    # the foot sensors' dropped readings draw from their own stream, so the pose noise, terrain
    # and shoves of a seed are what they were before the sensors existed
    ctrl = SkaterController(model, p, task, fans_enabled, sensor_rng=np.random.default_rng(seed + 7919),
                            lidar_rng=np.random.default_rng(seed + 104729))
    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
    # task key roll_score 'core': the ride-quality roll (roll_rms, grace) is the core's roll in the world, not the hip
    # box's (with a waist the two differ, and the core is what carries the electronics and fans); falls stay on the hip box
    core_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core") if task.get("roll_score") == "core" else -1
    pitch_max = math.radians(float(task["fall_pitch_deg"])) if task.get("fall_pitch_deg") else 1.0
    foot_bodies =[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{s}") for s in ("l", "r")]
    total_mass = float(model.body_subtreemass[hip_id])
    h_nom = nominal_height(p)
    # the robustness score's model errors (task mass_error / friction_error, fractions of the nominal, one
    # draw per run; a push: a shove of `push` x the stand's at a random moment of a legged run), drawn after
    # the controller has read the nominal model, so the controller is wrong by that much. Absent keys keep
    # every score and every seed's random stream
    if task.get("mass_error"):
        s_m = 1.0 + float(rng.uniform(-1.0, 1.0)) * float(task["mass_error"])
        model.body_mass[hip_id] *= s_m                # the dynamics read these directly; the derived
        model.body_inertia[hip_id] *= s_m             # subtree masses stay nominal, as the controller's
    if task.get("friction_error"):
        model.geom_friction[:, 0] *= 1.0 + float(rng.uniform(-1.0, 1.0)) * float(task["friction_error"])
    push_at = None
    if task.get("push") and objective != "stand":
        push_at = float(rng.uniform(1.0, max(1.0, float(task["duration"]) - 1.0)))
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    names = {g: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") for g in range(model.ngeom)}
    feet = {g for g, n in names.items() if n.startswith("foot_")}
    foot_side = {g: n[5] for g, n in names.items() if n.startswith("foot_")}      # geom -> "l" / "r"
    foot_z0 = {s: float(data.xpos[b][2]) for s, b in zip("lr", foot_bodies)}   # ankle height with the sole on the floor
    # the ground's height under a point: 0 except on the "slope" terrain (task slope_deg, slope_heading_deg), where the
    # fall height and the swing clearance are measured from the ground below, not from z = 0
    if task["terrain"] == "slope":
        _sd, _sh = float(task.get("slope_deg", 0.0)), float(task.get("slope_heading_deg", 0.0))
        ground = lambda x, y: float(terrain_mod.slope_height(x, y, _sd, _sh))
    else:
        ground = lambda x, y: 0.0
    swing_max = {"l": None, "r": None}            # highest ankle rise during the current swing
    clearances: list[tuple[float, float]] = []    # (time, rise) per swing, at touchdown
    cone_of = {g: int(n.split("_")[1]) for g, n in names.items() if n.startswith("cone_")}
    body_geoms = {g for g in range(model.ngeom) if g not in feet and g != floor_geom and g not in cone_of}
    cones_hit: set[int] = set()
    flights: list[tuple[float, float, float, float]] = []    # (apex rise, distance, duration, landing time)
    off_floor_since = None                            # time the feet left the floor, else None
    takeoff = None                                    # (x, z) at that moment
    apex = 0.0
    airborne_s = 0.0
    motor_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                  for n in ACTUATOR_NAMES if not n.startswith("fan_")]

    duration = float(task["duration"])
    n_ctrl = int(duration / CONTROL_DT)
    # per-step histories for the course's segment scores
    feet_hist, pitch_hist, cyc_hist = np.zeros(n_ctrl), np.zeros(n_ctrl), np.zeros(n_ctrl)
    sat_hist, acc_hist = np.zeros(n_ctrl), np.zeros(n_ctrl)
    # ... and for the walk's quality terms: body roll, hip height, forward hip speed, fan throttle
    roll_hist, z_hist, vx_hist, thr_hist = np.zeros(n_ctrl), np.zeros(n_ctrl), np.zeros(n_ctrl), np.zeros(n_ctrl)
    y_hist, track_hist = np.zeros(n_ctrl), np.zeros(n_ctrl)       # hip y; the feet's lateral separation (the track)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    feet_x0 = float(np.mean([data.xpos[b][0] for b in foot_bodies]))
    servo_prev: list[np.ndarray] = []                 # last two servo command vectors (chatter metric)
    servo_acc_sq, servo_sat = 0.0, 0
    servo_step = ctrl.servo.max_rad_s * CONTROL_DT
    # the scored saturation (servo_sat_frac, in the grace) has always measured every channel against
    # the leg servo's slew, which the roll channel (slewed at the roll servo's speed over the lever)
    # never reaches; the roll servo's own saturation is reported separately, not yet scored
    roll_step = ctrl.roll_servo.max_rad_s / ctrl.roll_ratio * CONTROL_DT
    roll_sat = 0
    # per-servo peak torque (fraction of stall) and speed (fraction of the no-load speed, at the
    # hinge) per gait cycle: whether the servos survive the gait, not just the slew metric
    servo_acts = ctrl.servo_acts
    servo_stall = np.array([model.actuator_forcerange[a][1] for a in servo_acts])
    servo_speed = np.array([ctrl.roll_servo.max_rad_s / ctrl.roll_ratio] * 2 + [ctrl.servo.max_rad_s] * 4)
    cyc_peak_t, cyc_peak_v = np.zeros(6), np.zeros(6)
    cycle_peaks_t: list[np.ndarray] = []
    cycle_peaks_v: list[np.ndarray] = []
    cycle_prev = int(ctrl.cycles)
    track_err_sq = np.zeros(3)                        # the ankles' error from the IK target, in the hip frame
    track_n = 0
    fall_roll = fall_pitch = 0.0
    fan_energy = motor_energy = 0.0
    time_upright = 0.0
    fell = unstable = False
    next_perturb, perturb_until = 1.0, -1.0
    accel_sq = 0.0
    lat_err_sq = 0.0
    scrub = 0.0
    v_prev = data.qvel[:3].copy()
    vel6 = np.zeros(6)
    steps = 0
    t_wall = time.perf_counter()
    A, lam = float(task.get("amplitude", 0.5)), float(task.get("wavelength", 3.0))

    for k in range(n_ctrl):
        t = k * CONTROL_DT
        data.ctrl[:] = ctrl.act(data, t)

        if objective == "stand":
            if t >= next_perturb:
                ang = rng.uniform(0, 2 * math.pi)
                f = PERTURB_ACCEL * total_mass
                data.xfrc_applied[hip_id, :3] = [f * math.cos(ang), f * math.sin(ang), 0.0]
                perturb_until = t + PERTURB_LEN
                next_perturb += PERTURB_PERIOD
            elif t >= perturb_until:
                data.xfrc_applied[hip_id, :] = 0.0
        elif push_at is not None and t >= push_at:
            if perturb_until < 0.0:
                ang = rng.uniform(0, 2 * math.pi)
                f = float(task["push"]) * PERTURB_ACCEL * total_mass
                data.xfrc_applied[hip_id, :3] = [f * math.cos(ang), f * math.sin(ang), 0.0]
                perturb_until = t + PERTURB_LEN
            elif t >= perturb_until:
                data.xfrc_applied[hip_id, :] = 0.0
                push_at = None

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        steps += SUBSTEPS

        if not np.all(np.isfinite(data.qpos)):
            unstable = True
            break

        thr_l, thr_r = ctrl.last_throttle
        fan_energy += (thr_l ** 2 + thr_r ** 2) * CONTROL_DT
        thr_hist[k] = 0.5 * (thr_l + thr_r)
        servo_cmd = np.array([data.ctrl[ctrl.idx[f"hip_roll_{s}"]] for s in "lr"]
                             + [data.ctrl[ctrl.idx[f"hip_pitch_{s}"]] for s in "lr"]
                             + [ctrl.horn_cmd[s] for s in "lr"]
                             + [data.ctrl[ctrl.idx[f"fan_pitch_{s}"]] for s in "lr"])
        if len(servo_prev) == 2:
            acc_hist[k] = float(np.mean((servo_cmd - 2 * servo_prev[-1] + servo_prev[-2]) ** 2))
            sat_hist[k] = int(np.any(np.abs(servo_cmd[:6] - servo_prev[-1][:6]) >= 0.98 * servo_step))
            roll_sat += int(np.any(np.abs(servo_cmd[:2] - servo_prev[-1][:2]) >= 0.98 * roll_step))
            servo_acc_sq += acc_hist[k]
            servo_sat += int(sat_hist[k])
        feet_hist[k] = float(np.mean([data.xpos[b][0] for b in foot_bodies]))
        track_hist[k] = abs(float(data.xpos[foot_bodies[0]][1]) - float(data.xpos[foot_bodies[1]][1]))
        cyc_hist[k] = ctrl.cycles
        servo_prev = (servo_prev + [servo_cmd])[-2:]
        motor_energy += float(np.sum(np.abs(data.actuator_force[motor_acts] * data.actuator_velocity[motor_acts]))) * CONTROL_DT
        cyc_peak_t = np.maximum(cyc_peak_t, np.abs(data.actuator_force[servo_acts]) / servo_stall)
        cyc_peak_v = np.maximum(cyc_peak_v, np.abs(data.actuator_velocity[servo_acts]) / servo_speed)
        if int(ctrl.cycles) != cycle_prev:
            cycle_peaks_t.append(cyc_peak_t); cycle_peaks_v.append(cyc_peak_v)
            cyc_peak_t, cyc_peak_v, cycle_prev = np.zeros(6), np.zeros(6), int(ctrl.cycles)
        R_hip = data.xmat[hip_id].reshape(3, 3)
        for s_, b in zip("lr", foot_bodies):
            track_err_sq += (R_hip.T @ (data.xpos[b] - data.xpos[hip_id]) - ctrl.foot_target[s_]) ** 2
        track_n += 2
        v = data.qvel[:3].copy()
        accel_sq += float(np.sum(((v - v_prev) / CONTROL_DT) ** 2))
        v_prev = v
        feet_down = False
        side_down = {"l": False, "r": False}
        for ci in range(data.ncon):
            g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
            if (g1 == floor_geom and g2 in feet) or (g2 == floor_geom and g1 in feet):
                feet_down = True
                side_down[foot_side[g2 if g1 == floor_geom else g1]] = True
            elif g1 in cone_of:
                cones_hit.add(cone_of[g1])
            elif g2 in cone_of:
                cones_hit.add(cone_of[g2])
        for s, b in zip("lr", foot_bodies):          # swing-foot clearance: the ankle's rise per swing
            rise = float(data.xpos[b][2]) - foot_z0[s] - ground(float(data.xpos[b][0]), float(data.xpos[b][1]))
            if not side_down[s]:
                swing_max[s] = rise if swing_max[s] is None else max(swing_max[s], rise)
            elif swing_max[s] is not None:
                clearances.append((t, swing_max[s]))
                swing_max[s] = None
        if not feet_down:
            if off_floor_since is None:
                off_floor_since, takeoff, apex = t, (float(data.qpos[0]), float(data.qpos[2])), float(data.qpos[2])
            apex = max(apex, float(data.qpos[2]))
            airborne_s += CONTROL_DT
        elif off_floor_since is not None:
            if t - off_floor_since >= FLIGHT_MIN_S and apex - takeoff[1] >= FLIGHT_MIN_RISE:
                flights.append((apex - takeoff[1], float(data.qpos[0]) - takeoff[0], t - off_floor_since, t))
            off_floor_since = None
        if objective == "slalom":
            y_ref = A * math.sin(2 * math.pi * float(data.qpos[0]) / lam)
            lat_err_sq += (float(data.qpos[1]) - y_ref) ** 2
            for b in foot_bodies:
                if data.xpos[b][2] < h_nom * 0.3:          # foot near the ground
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 1)
                    scrub += abs(float(vel6[4])) * CONTROL_DT   # local-frame lateral speed

        z = float(data.qpos[2]) - ground(float(data.qpos[0]), float(data.qpos[1]))
        roll, pitch = quat_to_roll_pitch(data.qpos[3:7])
        pitch_hist[k], roll_hist[k], z_hist[k], vx_hist[k], y_hist[k] = pitch, roll, z, float(v[0]), float(data.qpos[1])
        if core_roll_id >= 0:
            Rc = data.xmat[core_roll_id]
            roll_hist[k] = math.atan2(Rc[7], Rc[8])
        if _fallen(z, roll, pitch, h_nom, pitch_max) or _body_touching_floor(data, body_geoms, floor_geom):
            if not fell:
                fall_roll, fall_pitch = roll, pitch
            fell = True
            if stop_on_fall:
                break
        elif not fell:
            time_upright = t + CONTROL_DT

        if frame_cb is not None:
            frame_cb(model, data, t)
        if lidar_cb is not None and ctrl.lidar is not None and ctrl.lidar.new_frame:
            lidar_cb(ctrl.lidar, t)

    wall = time.perf_counter() - t_wall
    if np.any(cyc_peak_t > 0.0):                     # the unfinished last cycle
        cycle_peaks_t.append(cyc_peak_t); cycle_peaks_v.append(cyc_peak_v)
    peaks = _servo_peaks(cycle_peaks_t, cycle_peaks_v)
    track_rms = np.sqrt(track_err_sq / max(1, track_n))
    dx = float(data.qpos[0]) - x0
    dy = float(data.qpos[1]) - y0
    feet_dx = float(np.mean([data.xpos[b][0] for b in foot_bodies])) - feet_x0
    n = max(1, k + 1)
    rms_accel = math.sqrt(accel_sq / n)
    rms_lat = math.sqrt(lat_err_sq / n)
    upright_frac = time_upright / duration
    servo_acc_rms = math.sqrt(servo_acc_sq / max(1, n - 2)) / CONTROL_DT ** 2
    servo_sat_frac = servo_sat / max(1, n - 2)
    strides = p[GAIT_HZ.get(objective, "step_hz")] * max(time_upright, CONTROL_DT)
    stride = feet_dx / strides if objective in GAIT_HZ else 0.0
    cap_frac = RUN_STRIDE_CAP_FRAC if objective == "run" else STRIDE_CAP_FRAC
    stride_eff = max(0.0, min(stride, cap_frac * leg_reach(p)))
    # share of the upright time spent in real flights (both feet off >= FLIGHT_MIN_S with the hip
    # rising >= FLIGHT_MIN_RISE): a fast walk already loses foot contact a fifth of the time
    # without the hip ever rising, and that flicker is not a bound
    flight_frac = sum(f[2] for f in flights) / max(time_upright, CONTROL_DT)
    swing_clear = float(np.median([c for _, c in clearances])) if clearances else 0.0
    quality = _gait_quality(pitch_hist, roll_hist, z_hist, vx_hist, thr_hist, y_hist, track_hist, 0, max(2, min(n, int(time_upright / CONTROL_DT))))
    flight_len = float(np.mean([f[1] for f in flights])) if flights else 0.0
    course = (_course_scores(task, p, k + 1, time_upright, feet_hist, pitch_hist, cyc_hist, sat_hist, acc_hist,
                             flights, clearances, (roll_hist, z_hist, vx_hist, thr_hist, y_hist, track_hist)) if objective == "course" else {})
    clear_bonus = float(task.get("clear_weight", CLEAR_WEIGHT.get(objective, 0.0))) * max(0.0, min(swing_clear, CLEAR_CAP))
    leg_frac = leg_mass_fraction(p)
    # the config may price the distal mass itself (`leg_mass_penalty`, 0 to ignore it for a round)
    leg_pen = (LEG_MASS_PENALTY if leg_mass_penalty is None else leg_mass_penalty) * 2 * max(0.0, leg_frac - LEG_MASS_FRACTION_TARGET)
    energy = fan_energy_cost * fan_energy + motor_energy_cost * motor_energy
    accel_cost = float(task.get("accel_cost", ACCEL_COST.get(objective, 0.02)))
    apexes = sorted((f[0] for f in flights), reverse=True)
    max_apex = apexes[0] if apexes else 0.0
    max_flight = max([f[1] for f in flights], default=0.0)

    if unstable:
        fitness = UNSTABLE_FITNESS
    elif objective == "skate":
        fitness = (dx / duration) * upright_frac - accel_cost * rms_accel - 0.2 * abs(dy)
    elif objective == "walk":
        # grace over speed: the gait score scaled by how smoothly the body rides, the fans charged by duty
        gait_score = (float(task.get("speed_weight", WALK_SPEED_WEIGHT)) * feet_dx / duration
                      + float(task.get("stride_weight", WALK_STRIDE_WEIGHT)) * stride_eff)
        quality["grace"] = _grace(task, quality, servo_sat_frac, p["hip_width"])
        speed_gate = min(1.0, (feet_dx / duration) / float(task.get("speed_gate", WALK_SPEED_GATE)))
        stride_min = float(task.get("stride_min", 0.0))
        if stride_min > 0.0:
            speed_gate = min(speed_gate, max(0.0, stride) / stride_min)
        clear_min = float(task.get("clear_min", 0.0))
        if clear_min > 0.0:
            speed_gate = min(speed_gate, max(0.0, swing_clear) / clear_min)
        # the clearance bonus is gated by progress too: left outside the gate, the first track round
        # marched in place with its feet lifted 6 cm (2 cm of progress in 12 s) and won on clearance
        fitness = (gait_score * quality["grace"] * speed_gate + clear_bonus * speed_gate - accel_cost * rms_accel - 0.2 * abs(dy)
                   - float(task.get("fan_duty_cost", FAN_DUTY_COST["walk"])) * quality["fan_duty"] * upright_frac
                   - float(task.get("stand_cost", STAND_COST)) * (1.0 - speed_gate) * upright_frac)
    elif objective == "run":
        # speed of the feet, long strides and long flights, all gated by having real flights at all
        # (a bound, not a fast walk), the hop's bob charged
        gate = min(1.0, flight_frac / float(task.get("flight_gate", FLIGHT_GATE)))
        fitness = (gate * (float(task.get("speed_weight", RUN_SPEED_WEIGHT)) * feet_dx / duration
                           + float(task.get("stride_weight", 0.5)) * stride_eff
                           + float(task.get("flight_len_weight", FLIGHT_LEN_WEIGHT)) * min(flight_len, FLIGHT_LEN_CAP))
                   + float(task.get("flight_weight", RUN_FLIGHT_WEIGHT)) * min(flight_frac, FLIGHT_FRAC_CAP) + clear_bonus
                   - float(task.get("bob_cost", RUN_BOB_COST)) * quality["bob_m"] - accel_cost * rms_accel - 0.2 * abs(dy))
    elif objective == "course":
        scale = COST_REF_S / duration
        energy *= scale
        fitness = course["course_score"] - accel_cost * rms_accel - 0.2 * abs(dy) * scale
    elif objective == "hop":
        fitness = sum(apexes[:HOP_TOP_N]) - float(task.get("drift_cost", 0.3)) * math.hypot(dx, dy)
    elif objective == "broad_jump":
        fitness = max_flight + float(task.get("dist_weight", 0.1)) * dx
    elif objective == "crab":
        fitness = (dy / duration) * upright_frac - accel_cost * rms_accel - 0.2 * abs(dx)
    elif objective == "slalom":
        fitness = ((dx / duration) * upright_frac - float(task.get("lat_cost", 1.0)) * rms_lat
                   - float(task.get("scrub_cost", 0.5)) * scrub
                   - float(task.get("cone_cost", CONE_COST)) * len(cones_hit))
    else:
        fitness = upright_frac - float(task.get("drift_cost", 1.0)) * math.hypot(dx, dy)
    if not unstable:
        fall_cost = float(task.get("fall_cost", FALL_COST.get(objective, 0.5)))
        if objective in STEPPING:
            fall_cost *= 1.0 - upright_frac
        chatter = (float(task.get("chatter_cost", CHATTER_COST.get(objective, 0.0))) * servo_acc_rms
                   + float(task.get("sat_cost", SAT_COST.get(objective, 0.0))) * servo_sat_frac)
        fitness += -(fall_cost if fell else 0.0) - energy - leg_pen - chatter

    return {
        "fitness": float(fitness),
        "invalid": False,
        "distance_x": dx,
        "feet_x_m": feet_dx,
        "stride_m": stride,
        "flight_frac": flight_frac,
        "swing_clear_m": swing_clear,
        "servo_accel_rms": servo_acc_rms,
        **quality,
        "flight_len_m": flight_len,
        "apex_m": float(np.mean([f[0] for f in flights])) if flights else 0.0,
        "servo_sat_frac": servo_sat_frac,
        "drift_y": dy,
        "mean_speed": (dx if objective != "crab" else dy) / duration,
        "time_upright": time_upright,
        "fell": fell,
        "fall_dir": _fall_dir(fell, fall_roll, fall_pitch),
        "contact_events": dict(ctrl.contact_events),
        "catch_events": int(getattr(ctrl, "catch_events", 0)),
        "unstable": unstable,
        "rms_accel": rms_accel,
        "rms_lat_err": rms_lat,
        "scrub_m": scrub,
        "cones_hit": len(cones_hit),
        "cones": len(cone_of) // CONE_LAYERS,
        "n_flights": len(flights),
        "max_apex_m": max_apex,
        "max_flight_m": max_flight,
        "airborne_s": airborne_s,
        "fan_energy": fan_energy,
        "motor_energy_j": motor_energy,
        "leg_mass_fraction": leg_frac,
        "leg_mass_penalty": leg_pen,
        "total_mass": total_mass,
        **ctrl.sensors.summary(),
        **(ctrl.lidar.summary() if ctrl.lidar is not None else {}),
        "roll_servo_sat_frac": roll_sat / max(1, n - 2),
        **peaks,
        "foot_track_rms_m": float(np.linalg.norm(track_rms)),
        "foot_track_y_rms_m": float(track_rms[1]),
        "sim_steps": steps,
        "wall_s": wall,
        "steps_per_s": steps / wall if wall > 0 else 0.0,
        **course,
    }


def _fall_dir(fell: bool, roll: float, pitch: float) -> str:
    """Which way the body went at the fall: forward (nose down, pitch > 0), back, right (roll > 0,
    right side down) or left, whichever angle was larger; "drop" when the hip sank with neither
    past 10 deg; "none" standing."""
    if not fell:
        return "none"
    if max(abs(roll), abs(pitch)) < math.radians(10):
        return "drop"
    if abs(pitch) >= abs(roll):
        return "forward" if pitch > 0 else "back"
    return "right" if roll > 0 else "left"


def _servo_peaks(cycles_t: list, cycles_v: list) -> dict[str, float]:
    """Per-servo peak torque and speed per gait cycle, reduced per joint (the worse side) to the
    median cycle (`peak_torque_hip_roll`, ... fractions of stall / no-load speed) and the worst
    cycle of all six (`servo_peak_torque_frac`, `servo_peak_speed_frac`)."""
    if not cycles_t:
        return {f"peak_{k}_{j}": 0.0 for k in ("torque", "speed") for j in ("hip_roll", "hip_pitch", "knee")} | {"servo_peak_torque_frac": 0.0, "servo_peak_speed_frac": 0.0}
    T, V = np.array(cycles_t), np.array(cycles_v)             # (cycles, 6): roll l r, pitch l r, knee l r
    out = {}
    for k, M in (("torque", T), ("speed", V)):
        med = np.median(M, axis=0)
        for j, (a, b) in zip(("hip_roll", "hip_pitch", "knee"), ((0, 1), (2, 3), (4, 5))):
            out[f"peak_{k}_{j}"] = float(max(med[a], med[b]))
    out["servo_peak_torque_frac"] = float(T.max())
    out["servo_peak_speed_frac"] = float(V.max())
    return out


def _gait_quality(pitch, roll, z, vx, thr, y, track, i0: int, i1: int) -> dict:
    """How smoothly the body rides over [i0, i1): the sway (body pitch and roll RMS, rad), the
    bob (hip height standard deviation, m), the surge (forward hip speed standard deviation,
    m/s), the weave (lateral hip excursion about its WEAVE_WIN_S moving average, m), the
    fan duty (mean throttle of the two fans, 0-1) and the track (mean lateral separation of
    the feet, m). A graceful walk keeps all of them small; a speedy one need not."""
    if i1 - i0 < 2:
        return {"sway_rad": 0.0, "pitch_rms": 0.0, "roll_rms": 0.0, "bob_m": 0.0, "surge_ms": 0.0, "weave_m": 0.0, "fan_duty": 0.0, "track_m": 0.0}
    pr, rr = float(np.sqrt(np.mean(pitch[i0:i1] ** 2))), float(np.sqrt(np.mean(roll[i0:i1] ** 2)))
    yy = np.asarray(y[i0:i1], dtype=float)
    w = max(1, min(len(yy), int(round(WEAVE_WIN_S / CONTROL_DT))))
    trend = np.convolve(np.pad(yy, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    return {"sway_rad": pr + rr, "pitch_rms": pr, "roll_rms": rr, "bob_m": float(np.std(z[i0:i1])),
            "surge_ms": float(np.std(vx[i0:i1])), "weave_m": float(np.std(yy - trend)), "fan_duty": float(np.mean(thr[i0:i1])),
            "track_m": float(np.mean(track[i0:i1]))}


def _grace(task: dict, q: dict, sat_frac: float, track_free: float = 0.0) -> float:
    """The grace factor of a walk, 1 for a body that glides on quiet servos and 1/2 at one
    reference's worth of sway, bob, surge, weave, servo saturation or track beyond track_free
    (the hip width: feet under the hips are free, feet splayed outboard are charged); task keys
    sway_ref / bob_ref / surge_ref / weave_ref / sat_ref / track_ref move the references."""
    return 1.0 / (1.0 + max(0.0, q.get("track_m", 0.0) - track_free) / float(task.get("track_ref", TRACK_REF)) + q["sway_rad"] / float(task.get("sway_ref", SWAY_REF)) + q["bob_m"] / float(task.get("bob_ref", BOB_REF))
                  + q["surge_ms"] / float(task.get("surge_ref", SURGE_REF)) + q["weave_m"] / float(task.get("weave_ref", WEAVE_REF))
                  + sat_frac / float(task.get("sat_ref", SAT_REF)))


def _course_scores(task: dict, p: dict, n_done: int, time_upright: float, feet, pitch, cyc, sat, acc,
                   flights, clearances, body) -> dict:
    """Segment and transition scores of a course (objective "course").

    Each profile segment is scored on its steady window (the segment less the first
    TRANS_WIN_S after a gait change): a walk or run on the feet's speed, the realized
    stride (progress per cycle of the shared phase, capped as on the plain objectives),
    the share of the window in real flights (run) and the swing clearance, each with its
    gait's weights, weighted by the segment's share of the stepping time, a walk's scaled by
    its grace factor and charged its fan duty (as on the walk objective); a stand on the
    distance the feet move and the pitch RMS (the stop). Each gait change earns
    `trans_weight` x (1 - peak |pitch| / TRANS_PITCH) over the TRANS_WIN_S after it if
    the body is still upright at the end of the window, and reports the time to 90 % of
    the new segment's steady speed (the start). Segments after a fall score nothing, and
    the rollout's fall cost scales with the time left, so a course is worth finishing.
    Returns flat keys (seg<k>_<gait>.<metric>, trans<k>_<gait>.<metric>) and course_score."""
    dt = CONTROL_DT
    profile = task["profile"]
    leg = leg_reach(p)
    win, smooth = int(round(TRANS_WIN_S / dt)), int(round(T90_SMOOTH_S / dt))
    roll, z, vx, thr, y, track = body
    gate_ref, w_len = float(task.get("flight_gate", FLIGHT_GATE)), float(task.get("flight_len_weight", FLIGHT_LEN_WEIGHT))
    bob_cost = float(task.get("bob_cost", RUN_BOB_COST))
    speed_gate = float(task.get("speed_gate", WALK_SPEED_GATE))
    w_fan = float(task.get("fan_duty_cost", FAN_DUTY_COST["course"]))
    w_speed = {"walk": float(task.get("walk_speed_weight", COURSE_SPEED_WEIGHT["walk"])),
               "run": float(task.get("run_speed_weight", COURSE_SPEED_WEIGHT["run"]))}
    w_stride = {"walk": float(task.get("walk_stride_weight", COURSE_STRIDE_WEIGHT["walk"])),
                "run": float(task.get("run_stride_weight", COURSE_STRIDE_WEIGHT["run"]))}
    w_flight = float(task.get("flight_weight", COURSE_FLIGHT_WEIGHT))
    w_clear = {g: float(task.get("clear_weight", COURSE_CLEAR_WEIGHT[g])) for g in ("walk", "run")}
    w_trans, stop_cost = float(task.get("trans_weight", TRANS_WEIGHT)), float(task.get("stop_cost", STOP_COST))
    gait_total = sum(d for g, d in profile if g != "stand") or 1.0
    out, total = {}, 0.0
    t0 = 0.0
    for k, (g, d) in enumerate(profile):
        s, e = t0, t0 + d
        t0 = e
        i0, i1 = int(round(s / dt)), min(n_done, int(round(e / dt)))
        j0 = min(i1, i0 + win) if k > 0 else i0            # steady window: after the change settles
        n = i1 - j0
        seg = {"upright_s": max(0.0, min(time_upright, e) - s), "progress_m": 0.0, "speed": 0.0, "stride_m": 0.0,
               "flight_frac": 0.0, "flight_len_m": 0.0, "swing_clear_m": 0.0, "sat_frac": 0.0, "chatter_rms": 0.0, "grace": 1.0,
               **_gait_quality(pitch, roll, z, vx, thr, y, track, j0, i1)}
        if n > 1:
            T = n * dt
            prog = float(feet[i1 - 1] - feet[j0])
            cycles = float(cyc[i1 - 1] - cyc[j0])
            cl = [c for t_c, c in clearances if j0 * dt <= t_c < i1 * dt]
            fl = [f for f in flights if j0 * dt <= f[3] < i1 * dt]
            seg.update(progress_m=prog, speed=prog / T, stride_m=prog / cycles if cycles > 0.5 else 0.0,
                       flight_frac=sum(f[2] for f in fl) / T, flight_len_m=float(np.mean([f[1] for f in fl])) if fl else 0.0,
                       swing_clear_m=float(np.median(cl)) if cl else 0.0,
                       sat_frac=float(np.mean(sat[j0:i1])),
                       chatter_rms=float(np.sqrt(np.mean(acc[j0:i1]))) / dt ** 2)
            seg["grace"] = _grace(task, seg, seg["sat_frac"], p["hip_width"])
        if g == "stand":
            score = (-stop_cost * abs(seg["progress_m"]) - SETTLE_COST * seg["pitch_rms"]) if n > 1 else 0.0
        else:
            cap = (RUN_STRIDE_CAP_FRAC if g == "run" else STRIDE_CAP_FRAC) * leg
            gait_score = w_speed[g] * seg["speed"] + w_stride[g] * max(0.0, min(seg["stride_m"], cap))
            if g == "walk":
                gait_score = gait_score * seg["grace"] * min(1.0, seg["speed"] / speed_gate) - w_fan * seg["fan_duty"]
            else:                                            # the run: gated by flight, paid for flight length, charged the bob
                gait_score = (min(1.0, seg["flight_frac"] / gate_ref) * (gait_score + w_len * min(seg["flight_len_m"], FLIGHT_LEN_CAP))
                              + w_flight * min(seg["flight_frac"], FLIGHT_FRAC_CAP) - bob_cost * seg["bob_m"])
            clear = w_clear[g] * max(0.0, min(seg["swing_clear_m"], CLEAR_CAP))
            if g == "walk":
                clear *= min(1.0, seg["speed"] / speed_gate)        # no credit for marching in place
            score = (d / gait_total) * (gait_score + clear)
        seg["score"] = score
        total += score
        out.update({f"seg{k}_{g}.{m}": v for m, v in seg.items()})
        if k > 0:                                            # the change into this segment
            i_end = min(n_done, i0 + win)
            peak = float(np.max(np.abs(pitch[i0:i_end]))) if i_end > i0 else 0.0
            survived = time_upright >= min(e, s + TRANS_WIN_S) - 1e-9
            t90 = d
            if g != "stand" and seg["speed"] > 0.05:
                for i in range(i0, i1 - smooth):
                    if (feet[i + smooth] - feet[i]) / (smooth * dt) >= 0.9 * seg["speed"]:
                        t90 = (i + smooth) * dt - s
                        break
            tr = {"survived": float(survived), "pitch_max_deg": math.degrees(peak), "t90_s": t90,
                  "score": w_trans * max(0.0, 1.0 - peak / TRANS_PITCH) if survived else 0.0}
            total += tr["score"]
            out.update({f"trans{k}_{g}.{m}": v for m, v in tr.items()})
    out["course_score"] = total
    out["course_done"] = float(time_upright >= t0 - 1e-9)
    return out


def evaluate(genome: np.ndarray, cfg: dict, seed: int) -> dict:
    """Weighted sum of the tasks' fitness. A task with `repeats` n is rolled out n times on
    different seeds (its pose noise, terrain, shoves and model errors differ) and scores the
    mean of them, or with `aggregate` "pNN" the NN-th percentile and "min" the worst (the
    robustness score: the mean of two is a knife edge a GA walks along, and twice it has
    handed back a winner worse than its seed on four tilted starts); the metrics are the
    worst-upright run's."""
    out = {"fitness": 0.0, "sim_steps": 0, "wall_s": 0.0, "invalid": False}
    for ti, task in enumerate(cfg["tasks"]):
        runs = [rollout(genome, task, seed=seed * 1000 + ti + 100 * rep,
                        fans_enabled=cfg.get("fans_enabled", True),
                        fan_energy_cost=cfg.get("fan_energy_cost", 0.2),
                        motor_energy_cost=cfg.get("motor_energy_cost", 0.002),
                        leg_mass_penalty=cfg.get("leg_mass_penalty"))
                for rep in range(int(task.get("repeats", 1)))]
        fits = [r["fitness"] for r in runs]
        agg = str(task.get("aggregate", "mean"))
        score = (float(np.mean(fits)) if agg == "mean" else float(min(fits)) if agg == "min"
                 else float(np.percentile(fits, float(agg[1:]))))
        m = min(runs, key=lambda r: (r["time_upright"], r["fitness"]))
        m = {**m, "fitness": score, "fitness_mean": float(np.mean(fits)), "fitness_min": float(min(fits)),
             "upright_min": float(min(r["time_upright"] for r in runs)),
             "sim_steps": sum(r["sim_steps"] for r in runs), "wall_s": sum(r["wall_s"] for r in runs)}
        out["fitness"] += task.get("weight", 1.0) * m["fitness"]
        out["sim_steps"] += m["sim_steps"]
        out["wall_s"] += m["wall_s"]
        out["invalid"] = out["invalid"] or m["invalid"]
        for k2, v in m.items():
            if not isinstance(v, (list, dict)):
                out[f"{task['name']}.{k2}"] = v
        if m["invalid"]:
            break
    out["steps_per_s"] = out["sim_steps"] / out["wall_s"] if out["wall_s"] > 0 else 0.0
    return out
