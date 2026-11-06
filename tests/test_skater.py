import json

import mujoco
import numpy as np
import pytest

from evol_mech import skater
from evol_mech.bodies import get_body
from evol_mech.skater import (DEFAULT_NORM, N_PARAMS, decode, encode, random_genome, mutate, crossover,
                              structure_from_params, make_model, make_data, rollout, evaluate, InvalidBody)
from evol_mech.skater.structure import (nominal_angles, foot_from_angles, angles_from_foot, foot_layout,
                                        LEG_MASS_FRACTION_TARGET)
from evol_mech.skater.mjcf import ACTUATOR_NAMES
from evol_mech.skater.rollout import LEG_MASS_PENALTY


def test_roundtrip_and_operators():
    assert np.allclose(encode(decode(DEFAULT_NORM)), DEFAULT_NORM, atol=1e-9)
    rng = np.random.default_rng(1)
    for _ in range(20):
        c = mutate(crossover(random_genome(rng), random_genome(rng), rng), rng, 0.3, 0.5)
        assert c.shape == (N_PARAMS,) and c.min() >= 0.0 and c.max() <= 1.0


def test_nominal_crouch_and_two_link_ik():
    p = decode(DEFAULT_NORM)
    thigh, shin = nominal_angles(p)
    assert abs(shin - thigh - p["crouch"]) < 1e-9             # knee flexion = crouch
    x, z = foot_from_angles(p, thigh, shin)
    assert abs(x) < 1e-9 and z < 0                             # ankle straight under the hip
    # both servos move for a pure height change; one servo alone swings the foot on an arc
    t2, s2 = angles_from_foot(p, 0.0, z + 0.03)
    assert abs(t2 - thigh) > 1e-3 and abs(s2 - shin) > 1e-3
    for xf, zf in ((0.0, z), (0.04, z + 0.02), (-0.05, z - 0.01)):
        t, s = angles_from_foot(p, xf, zf)
        assert np.allclose(foot_from_angles(p, t, s), (xf, zf), atol=1e-6)


def test_foot_layout_tread_and_wheels():
    p = decode(DEFAULT_NORM)
    lay = foot_layout(p)
    xs = np.array(lay.roller_x)
    assert len(xs) >= 2 and np.all(np.diff(xs) >= 2 * lay.roller_r)   # rollers never overlap
    assert abs(xs[0] + xs[-1]) < 1e-9                                  # symmetric about the ankle
    assert lay.idler is not None and lay.idler[0] > xs[-1] and lay.idler[1] > 0   # raised front idler
    w = foot_layout({**p, "foot_type": 1.0})
    assert len(w.roller_x) == 2 and w.roller_r > lay.roller_r and w.idler is None
    with pytest.raises(InvalidBody):
        foot_layout({**p, "sole_len": 0.08, "pulley_radius": 0.035})


def test_masses_follow_design_rules():
    p = decode(DEFAULT_NORM)
    st = structure_from_params(p)
    # all six servos live in the hip: the hip outweighs both legs together
    assert st.hip_kg > 2 * (st.thigh_kg + st.shin_kg + st.foot_kg)
    heavier = structure_from_params({**p, "servo_kgcm": 60})
    assert heavier.hip_kg > st.hip_kg and heavier.thigh_kg == st.thigh_kg
    # the leg-mass rule is enforced as a fitness penalty above the target
    r = rollout(encode({**p, "tread_motor_w": 150.0}), {"name": "stand", "terrain": "flat", "objective": "stand",
                                                         "duration": 0.5}, 0)
    assert r["leg_mass_fraction"] > LEG_MASS_FRACTION_TARGET
    assert abs(r["leg_mass_penalty"] - LEG_MASS_PENALTY * 2 * (r["leg_mass_fraction"] - LEG_MASS_FRACTION_TARGET)) < 1e-9


def test_model_builds_with_expected_actuators_and_contacts():
    model, p = make_model(DEFAULT_NORM, "flat", np.random.default_rng(0))
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)}
    assert names == set(ACTUATOR_NAMES)
    assert not any(n.startswith("ankle") for n in names)      # the ankle is passive
    ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_l")
    assert model.jnt_stiffness[ankle] == pytest.approx(p["ankle_stiffness"])
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    data = make_data(model, p)
    for _ in range(200):
        mujoco.mj_step(model, data)
    touching = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1 if c.geom2 == floor else c.geom2)
                for c in data.contact[:data.ncon] if floor in (c.geom1, c.geom2)}
    assert touching and all(n.startswith("foot_") for n in touching)
    # links, tread drive, fans and payload are reachable by the floor only
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not (n.startswith("foot_") or n in ("hip", "core", "floor")):
            assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 2


def test_tread_drives_the_body_forward_and_locks_at_zero():
    model, p = make_model(DEFAULT_NORM, "flat", np.random.default_rng(0))
    thigh, shin = nominal_angles(p)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def run(w):
        d = make_data(model, p)
        for s in ("l", "r"):
            d.ctrl[aid(f"hip_pitch_{s}")], d.ctrl[aid(f"knee_{s}")] = thigh, shin - thigh
            d.ctrl[aid(f"tread_{s}")] = w
        for _ in range(500):
            mujoco.mj_step(model, d)
        return float(d.qpos[0])
    assert abs(run(0.0)) < 0.01                         # locked tread: a parked sole
    assert run(8.0) > 0.08                              # positive roller rate rolls the body forward


def test_all_objectives_run_and_default_stands():
    for obj, extra in (("skate", {"speed": 0.6}), ("crab", {"speed": 0.3}), ("slalom", {"speed": 0.6}), ("stand", {})):
        r = rollout(DEFAULT_NORM, {"name": obj, "terrain": "flat", "objective": obj, "duration": 1.0, **extra}, 0)
        assert np.isfinite(r["fitness"]) and not r["invalid"]
    r = rollout(DEFAULT_NORM, {"name": "stand", "terrain": "flat", "objective": "stand", "duration": 3.0}, 0)
    assert r["time_upright"] >= 2.5 and not r["fell"]
    r = rollout(DEFAULT_NORM, {"name": "skate", "terrain": "flat", "objective": "skate", "speed": 0.6, "duration": 4.0}, 0)
    assert not r["fell"] and r["distance_x"] > 0.5


def test_registry_and_evaluate():
    body = get_body("skater")
    assert body.N_PARAMS == N_PARAMS and body.decode is decode
    cfg = json.load(open("configs/skater.json"))
    cfg["tasks"] = [dict(t, duration=0.5) for t in cfg["tasks"][:2]]
    out = body.evaluate(DEFAULT_NORM, cfg, seed=0)
    assert np.isfinite(out["fitness"]) and "skate_flat.mean_speed" in out
    with pytest.raises(ValueError):
        get_body("hexapod")
