"""Single-episode simulation of one skater genome on one task.

Task dict: {"name", "terrain", "objective": "skate" | "crab" | "slalom" | "stand",
"duration", "weight", "speed" (m/s, skate/crab/slalom), "amplitude" and
"wavelength" (slalom path y = A sin(2 pi x / lambda))}.

Fitness (stage-1 tasks from the design doc):
  skate   mean speed along +x, scaled by the fraction of time upright,
          minus ride-quality cost (RMS hip acceleration) and lateral drift
  crab    the same along +y, drift along x penalised
  slalom  mean speed along +x minus lateral path error and foot scrub
  stand   time upright minus drift, shoved every PERTURB_PERIOD seconds
All tasks: -0.5 for a fall, fan energy, leg + tread motor energy, and the
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

OBJECTIVES = ("skate", "crab", "slalom", "stand")
LEG_MASS_PENALTY = 5.0          # fitness per unit of leg fraction above target, per leg


def make_model(genome: np.ndarray, terrain_kind: str, rng: np.random.Generator):
    """Compile a model. Raises InvalidBody for un-buildable genomes."""
    p = decode(genome)
    asset, geom, hdata = terrain_mod.build(terrain_kind, rng)
    xml = build_mjcf(p, asset, geom, TERRAIN_ZMAX.get(terrain_kind, 0.0))
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
        model, p = make_model(genome, task["terrain"], rng)
    except InvalidBody as e:
        return _invalid_result(str(e))
    data = make_data(model, p)
    ctrl = SkaterController(model, p, task, fans_enabled)
    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{s}") for s in ("l", "r")]
    total_mass = float(model.body_subtreemass[hip_id])
    h_nom = nominal_height(p)
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = {g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("foot_")}
    body_geoms = {g for g in range(model.ngeom) if g not in feet and g != floor_geom}
    motor_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                  for n in ACTUATOR_NAMES if not n.startswith("fan_")]

    duration = float(task["duration"])
    n_ctrl = int(duration / CONTROL_DT)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
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
        motor_energy += float(np.sum(np.abs(data.actuator_force[motor_acts] * data.actuator_velocity[motor_acts]))) * CONTROL_DT
        v = data.qvel[:3].copy()
        accel_sq += float(np.sum(((v - v_prev) / CONTROL_DT) ** 2))
        v_prev = v
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
    n = max(1, k + 1)
    rms_accel = math.sqrt(accel_sq / n)
    rms_lat = math.sqrt(lat_err_sq / n)
    upright_frac = time_upright / duration
    leg_frac = leg_mass_fraction(p)
    leg_pen = LEG_MASS_PENALTY * 2 * max(0.0, leg_frac - LEG_MASS_FRACTION_TARGET)
    energy = fan_energy_cost * fan_energy + motor_energy_cost * motor_energy
    accel_cost = float(task.get("accel_cost", 0.02))

    if unstable:
        fitness = UNSTABLE_FITNESS
    elif objective == "skate":
        fitness = (dx / duration) * upright_frac - accel_cost * rms_accel - 0.2 * abs(dy)
    elif objective == "crab":
        fitness = (dy / duration) * upright_frac - accel_cost * rms_accel - 0.2 * abs(dx)
    elif objective == "slalom":
        fitness = ((dx / duration) * upright_frac - float(task.get("lat_cost", 1.0)) * rms_lat
                   - float(task.get("scrub_cost", 0.5)) * scrub)
    else:
        fitness = upright_frac - float(task.get("drift_cost", 1.0)) * math.hypot(dx, dy)
    if not unstable:
        fitness += -(0.5 if fell else 0.0) - energy - leg_pen

    return {
        "fitness": float(fitness),
        "invalid": False,
        "distance_x": dx,
        "drift_y": dy,
        "mean_speed": (dx if objective != "crab" else dy) / duration,
        "time_upright": time_upright,
        "fell": fell,
        "unstable": unstable,
        "rms_accel": rms_accel,
        "rms_lat_err": rms_lat,
        "scrub_m": scrub,
        "fan_energy": fan_energy,
        "motor_energy_j": motor_energy,
        "leg_mass_fraction": leg_frac,
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
