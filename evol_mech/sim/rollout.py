"""Single-episode simulation of one genome on one task.

A task is a dict: {"name": str, "terrain": str, "objective": "walk" | "stand",
"duration": seconds, "weight": float}. Returns a flat metrics dict including
`fitness` and throughput numbers for monitoring. No large arrays: workers
return this over multiprocessing, so it stays small.
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from ..body.mjcf import build_mjcf, nominal_leg_len, InvalidBody, SIM_DT
from ..body import linkage
from ..genome.spec import decode
from .controller import Controller, quat_to_roll_pitch
from . import terrain as terrain_mod

CONTROL_DT = 0.01                      # 100 Hz controller
SUBSTEPS = int(round(CONTROL_DT / SIM_DT))
INVALID_FITNESS = -5.0
UNSTABLE_FITNESS = -10.0

# "stand" task: horizontal shove every PERTURB_PERIOD seconds, lasting PERTURB_LEN
PERTURB_PERIOD = 1.5
PERTURB_LEN = 0.1
PERTURB_ACCEL = 4.0                    # m/s^2 equivalent, scaled by total mass

TERRAIN_ZMAX = {"flat": 0.0, "cobblestone": 0.02, "potholes": 0.03, "gravel": 0.012}


def make_model(genome: np.ndarray, terrain_kind: str, rng: np.random.Generator,
               park: bool = False, waist_yaw: bool = False):
    """Compile a model. Raises InvalidBody for un-buildable genomes."""
    p = decode(genome)
    asset, geom, hdata = terrain_mod.build(terrain_kind, rng)
    theta_l = linkage.parking_angle(p, p["crank_phase_r"]) if park else 0.0
    xml = build_mjcf(p, asset, geom, TERRAIN_ZMAX.get(terrain_kind, 0.0), theta_l=theta_l, waist_yaw=waist_yaw)
    model = mujoco.MjModel.from_xml_string(xml)
    if hdata is not None:
        model.hfield_data[:] = hdata.ravel()
    return model, p


def _fallen(z: float, roll: float, pitch: float, leg_len: float) -> bool:
    return z < 0.45 * leg_len or abs(roll) > 1.0 or abs(pitch) > 1.0


def _body_touching_floor(data: mujoco.MjData, body_geoms: set[int], floor_geom: int) -> bool:
    """True if the hip or core (anything but feet) is in contact with the floor.
    Sitting down must count as a fall, or evolution will learn to sit."""
    for k in range(data.ncon):
        c = data.contact[k]
        if (c.geom1 == floor_geom and c.geom2 in body_geoms) or            (c.geom2 == floor_geom and c.geom1 in body_geoms):
            return True
    return False


def _invalid_result(reason: str) -> dict:
    return {"fitness": INVALID_FITNESS, "invalid": True, "reason": reason,
            "distance_x": 0.0, "drift_y": 0.0, "time_upright": 0.0, "fell": True,
            "unstable": False, "fan_energy": 0.0, "total_mass": 0.0,
            "sim_steps": 0, "wall_s": 0.0, "steps_per_s": 0.0}


def rollout(genome: np.ndarray, task: dict, seed: int = 0, *,
            fans_enabled: bool = True, fan_energy_cost: float = 0.02, waist_yaw: bool = False,
            frame_cb=None, stop_on_fall: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    objective = task["objective"]
    try:
        model, p = make_model(genome, task["terrain"], rng, park=(objective == "stand"), waist_yaw=waist_yaw)
    except InvalidBody as e:
        return _invalid_result(str(e))
    data = mujoco.MjData(model)
    ctrl = Controller(model, p, fans_enabled, gait_enabled=(objective == "walk"))
    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
    total_mass = float(model.body_subtreemass[hip_id])
    leg_len = nominal_leg_len(p)
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    body_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("hip", "core")}

    duration = float(task["duration"])
    n_ctrl = int(duration / CONTROL_DT)
    mujoco.mj_forward(model, data)

    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    fan_energy = 0.0
    time_upright = 0.0
    fell = False
    unstable = False
    next_perturb = 1.0
    perturb_until = -1.0
    steps = 0
    t_wall = time.perf_counter()

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

        z = float(data.qpos[2])
        roll, pitch = quat_to_roll_pitch(data.qpos[3:7])
        if _fallen(z, roll, pitch, leg_len) or _body_touching_floor(data, body_geoms, floor_geom):
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

    if unstable:
        fitness = UNSTABLE_FITNESS
    elif objective == "walk":
        fitness = dx - 0.2 * abs(dy) - (0.5 if fell else 0.0) - fan_energy_cost * fan_energy
    elif objective == "stand":
        # drift_cost per metre: sliding away under the fans is not standing
        fitness = (time_upright / duration) - float(task.get("drift_cost", 1.0)) * math.hypot(dx, dy) - fan_energy_cost * fan_energy
    else:
        raise ValueError(f"unknown objective {objective!r}")

    return {
        "fitness": float(fitness),
        "invalid": False,
        "distance_x": dx,
        "drift_y": dy,
        "time_upright": time_upright,
        "fell": fell,
        "unstable": unstable,
        "fan_energy": fan_energy,
        "total_mass": total_mass,
        "sim_steps": steps,
        "wall_s": wall,
        "steps_per_s": steps / wall if wall > 0 else 0.0,
    }


def evaluate(genome: np.ndarray, cfg: dict, seed: int) -> dict:
    """Run every task in cfg['tasks'] and combine by weight. Returns the
    aggregate plus per-task metrics prefixed by task name."""
    out = {"fitness": 0.0, "sim_steps": 0, "wall_s": 0.0, "invalid": False}
    for ti, task in enumerate(cfg["tasks"]):
        m = rollout(genome, task, seed=seed * 1000 + ti,
                    fans_enabled=cfg.get("fans_enabled", True),
                    fan_energy_cost=cfg.get("fan_energy_cost", 0.02),
                    waist_yaw=cfg.get("waist_yaw", False))
        out["fitness"] += task.get("weight", 1.0) * m["fitness"]
        out["sim_steps"] += m["sim_steps"]
        out["wall_s"] += m["wall_s"]
        out["invalid"] = out["invalid"] or m["invalid"]
        for k, v in m.items():
            out[f"{task['name']}.{k}"] = v
        if m["invalid"]:
            break   # same body fails every task; don't waste time
    out["steps_per_s"] = out["sim_steps"] / out["wall_s"] if out["wall_s"] > 0 else 0.0
    return out
