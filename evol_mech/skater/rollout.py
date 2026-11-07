"""Single-episode simulation of one skater genome on one task.

Task dict: {"name", "terrain", "objective": "skate" | "crab" | "slalom" | "stand"
| "walk" | "hop" | "broad_jump", "duration", "weight", "speed" (m/s, skate/crab/
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
from .structure import InvalidBody, nominal_height, leg_mass_fraction, LEG_MASS_FRACTION_TARGET
from .controller import SkaterController

OBJECTIVES = ("skate", "crab", "slalom", "stand", "walk", "hop", "broad_jump")
STRIDE_CAP_FRAC = 0.75          # realized stride rewarded up to this fraction of thigh + shin (human step/leg ~0.7-0.8)
CHATTER_COST = {"walk": 1e-4}   # fitness per rad/s^2 of servo command acceleration RMS (0 elsewhere)
SAT_COST = {"walk": 0.1}        # fitness x the fraction of control steps with a leg servo at its rate limit
FALL_COST = {"walk": 0.1}       # default fall penalty per objective (0.5 otherwise); on the walk it
                                # scales with the time left, so an early fall pays it all and a late one little
ACCEL_COST = {"walk": 0.0}      # default ride-quality cost per objective (0.02 otherwise): a walk's
                                # step impacts are not a ride-quality fault, and at 0.02 they outweighed
                                # the progress of a 0.1 m/s walk, so standing still won
FLIGHT_MIN_S = 0.04             # both feet off the floor at least this long ...
FLIGHT_MIN_RISE = 0.01          # ... with the hip rising at least this (m) counts as a flight
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


def make_model(genome: np.ndarray, terrain_kind: str, rng: np.random.Generator, task: dict | None = None):
    """Compile a model. Raises InvalidBody for un-buildable genomes. With a slalom
    task the cones are placed in the scene."""
    p = decode(genome)
    if task and "step_hz_max" in task:            # a task may cap the cadence (slower, longer steps)
        p["step_hz"] = min(p["step_hz"], float(task["step_hz_max"]))
    asset, geom, hdata = terrain_mod.build(terrain_kind, rng)
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


def _fallen(z: float, roll: float, pitch: float, h_nom: float) -> bool:
    return z < 0.45 * h_nom or abs(roll) > 1.0 or abs(pitch) > 1.0


def rollout(genome: np.ndarray, task: dict, seed: int = 0, *,
            fans_enabled: bool = True, fan_energy_cost: float = 0.2,
            motor_energy_cost: float = 0.002, frame_cb=None, stop_on_fall: bool = True,
            **_ignored) -> dict:
    rng = np.random.default_rng(seed)
    objective = task["objective"]
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")
    try:
        model, p = make_model(genome, task["terrain"], rng, task)
    except InvalidBody as e:
        return _invalid_result(str(e))
    data = make_data(model, p)
    ctrl = SkaterController(model, p, task, fans_enabled)
    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{s}") for s in ("l", "r")]
    total_mass = float(model.body_subtreemass[hip_id])
    h_nom = nominal_height(p)
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    names = {g: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") for g in range(model.ngeom)}
    feet = {g for g, n in names.items() if n.startswith("foot_")}
    cone_of = {g: int(n.split("_")[1]) for g, n in names.items() if n.startswith("cone_")}
    body_geoms = {g for g in range(model.ngeom) if g not in feet and g != floor_geom and g not in cone_of}
    cones_hit: set[int] = set()
    flights: list[tuple[float, float, float]] = []    # (apex rise, distance, duration)
    off_floor_since = None                            # time the feet left the floor, else None
    takeoff = None                                    # (x, z) at that moment
    apex = 0.0
    airborne_s = 0.0
    motor_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                  for n in ACTUATOR_NAMES if not n.startswith("fan_")]

    duration = float(task["duration"])
    n_ctrl = int(duration / CONTROL_DT)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    feet_x0 = float(np.mean([data.xpos[b][0] for b in foot_bodies]))
    servo_prev: list[np.ndarray] = []                 # last two servo command vectors (chatter metric)
    servo_acc_sq, servo_sat = 0.0, 0
    servo_step = ctrl.servo.max_rad_s * CONTROL_DT
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

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        steps += SUBSTEPS

        if not np.all(np.isfinite(data.qpos)):
            unstable = True
            break

        thr_l, thr_r = ctrl.last_throttle
        fan_energy += (thr_l ** 2 + thr_r ** 2) * CONTROL_DT
        servo_cmd = np.array([data.ctrl[ctrl.idx[f"hip_roll_{s}"]] for s in "lr"]
                             + [data.ctrl[ctrl.idx[f"hip_pitch_{s}"]] for s in "lr"]
                             + [ctrl.horn_cmd[s] for s in "lr"]
                             + [data.ctrl[ctrl.idx[f"fan_pitch_{s}"]] for s in "lr"])
        if len(servo_prev) == 2:
            servo_acc_sq += float(np.mean((servo_cmd - 2 * servo_prev[-1] + servo_prev[-2]) ** 2))
            servo_sat += int(np.any(np.abs(servo_cmd[:6] - servo_prev[-1][:6]) >= 0.98 * servo_step))
        servo_prev = (servo_prev + [servo_cmd])[-2:]
        motor_energy += float(np.sum(np.abs(data.actuator_force[motor_acts] * data.actuator_velocity[motor_acts]))) * CONTROL_DT
        v = data.qvel[:3].copy()
        accel_sq += float(np.sum(((v - v_prev) / CONTROL_DT) ** 2))
        v_prev = v
        feet_down = False
        for ci in range(data.ncon):
            g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
            if (g1 == floor_geom and g2 in feet) or (g2 == floor_geom and g1 in feet):
                feet_down = True
            elif g1 in cone_of:
                cones_hit.add(cone_of[g1])
            elif g2 in cone_of:
                cones_hit.add(cone_of[g2])
        if not feet_down:
            if off_floor_since is None:
                off_floor_since, takeoff, apex = t, (float(data.qpos[0]), float(data.qpos[2])), float(data.qpos[2])
            apex = max(apex, float(data.qpos[2]))
            airborne_s += CONTROL_DT
        elif off_floor_since is not None:
            if t - off_floor_since >= FLIGHT_MIN_S and apex - takeoff[1] >= FLIGHT_MIN_RISE:
                flights.append((apex - takeoff[1], float(data.qpos[0]) - takeoff[0], t - off_floor_since))
            off_floor_since = None
        if objective == "slalom":
            y_ref = A * math.sin(2 * math.pi * float(data.qpos[0]) / lam)
            lat_err_sq += (float(data.qpos[1]) - y_ref) ** 2
            for b in foot_bodies:
                if data.xpos[b][2] < h_nom * 0.3:          # foot near the ground
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 1)
                    scrub += abs(float(vel6[4])) * CONTROL_DT   # local-frame lateral speed

        z = float(data.qpos[2])
        roll, pitch = quat_to_roll_pitch(data.qpos[3:7])
        if _fallen(z, roll, pitch, h_nom) or _body_touching_floor(data, body_geoms, floor_geom):
            fell = True
            if stop_on_fall:
                break
        elif not fell:
            time_upright = t + CONTROL_DT

        if frame_cb is not None:
            frame_cb(model, data, t)

    wall = time.perf_counter() - t_wall
    dx = float(data.qpos[0]) - x0
    dy = float(data.qpos[1]) - y0
    feet_dx = float(np.mean([data.xpos[b][0] for b in foot_bodies])) - feet_x0
    n = max(1, k + 1)
    rms_accel = math.sqrt(accel_sq / n)
    rms_lat = math.sqrt(lat_err_sq / n)
    upright_frac = time_upright / duration
    servo_acc_rms = math.sqrt(servo_acc_sq / max(1, n - 2)) / CONTROL_DT ** 2
    servo_sat_frac = servo_sat / max(1, n - 2)
    strides = p["step_hz"] * max(time_upright, CONTROL_DT)
    stride = feet_dx / strides if objective == "walk" else 0.0
    stride_eff = max(0.0, min(stride, STRIDE_CAP_FRAC * (p["thigh_len"] + p["shin_len"])))
    leg_frac = leg_mass_fraction(p)
    leg_pen = LEG_MASS_PENALTY * 2 * max(0.0, leg_frac - LEG_MASS_FRACTION_TARGET)
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
        fitness = (feet_dx / duration + float(task.get("stride_weight", 1.0)) * stride_eff
                   - accel_cost * rms_accel - 0.2 * abs(dy))
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
        if objective == "walk":
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
        "servo_accel_rms": servo_acc_rms,
        "servo_sat_frac": servo_sat_frac,
        "drift_y": dy,
        "mean_speed": (dx if objective != "crab" else dy) / duration,
        "time_upright": time_upright,
        "fell": fell,
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
        "sim_steps": steps,
        "wall_s": wall,
        "steps_per_s": steps / wall if wall > 0 else 0.0,
    }


def evaluate(genome: np.ndarray, cfg: dict, seed: int) -> dict:
    out = {"fitness": 0.0, "sim_steps": 0, "wall_s": 0.0, "invalid": False}
    for ti, task in enumerate(cfg["tasks"]):
        m = rollout(genome, task, seed=seed * 1000 + ti,
                    fans_enabled=cfg.get("fans_enabled", True),
                    fan_energy_cost=cfg.get("fan_energy_cost", 0.2),
                    motor_energy_cost=cfg.get("motor_energy_cost", 0.002))
        out["fitness"] += task.get("weight", 1.0) * m["fitness"]
        out["sim_steps"] += m["sim_steps"]
        out["wall_s"] += m["wall_s"]
        out["invalid"] = out["invalid"] or m["invalid"]
        for k2, v in m.items():
            out[f"{task['name']}.{k2}"] = v
        if m["invalid"]:
            break
    out["steps_per_s"] = out["sim_steps"] / out["wall_s"] if out["wall_s"] > 0 else 0.0
    return out
