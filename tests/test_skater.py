import math
import json

import mujoco
import numpy as np
import pytest

from evol_mech import skater
from evol_mech.bodies import get_body
from evol_mech.skater import (DEFAULT_NORM, N_PARAMS, decode, encode, random_genome, mutate, crossover,
                              structure_from_params, make_model, make_data, rollout, evaluate, InvalidBody)
from evol_mech.skater.structure import (nominal_angles, foot_from_angles, angles_from_foot, foot_layout,
                                        LEG_MASS_FRACTION_TARGET, shin_from_horn, horn_from_shin,
                                        drive_pose_allowed, drive_stops, coupler_len, KNEE_RANGE, FOLD_MARGIN)
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


def test_shin_drive_four_bar():
    import math
    p = decode(DEFAULT_NORM)
    thigh, shin = nominal_angles(p)
    # auto coupler: the horn is parallel to the shin at the nominal crouch, so horn = shin there
    horn = horn_from_shin(p, thigh, shin)
    assert horn is not None and abs(horn - shin) < 1e-6
    # forward and inverse agree away from nominal
    for t, h in ((-0.9, 0.7), (-0.3, 1.0), (-0.1, 0.2)):
        r = shin_from_horn(p, t, h)
        assert r is not None
        back = horn_from_shin(p, t, r[0])
        assert back is not None and abs(back - h) < 1e-5
    # the staggered drive couples the shin to the thigh: same horn, different thigh -> different shin
    s1 = shin_from_horn(p, thigh, horn)[0]
    s2 = shin_from_horn(p, thigh - 0.5, horn)[0]
    assert abs(s1 - s2) > 0.02
    # coaxial servo, equal horn / crank, coupler = thigh: LEO's parallelogram, shin = horn for any thigh
    para = {"dx": 0.0, "dz": 0.0, "horn": 0.03, "crank": 0.03, "crank_angle": 0.0, "coupler": p["thigh_len"]}
    assert abs(coupler_len(p, para) - p["thigh_len"]) < 1e-12
    for t in (-1.0, -0.3, 0.2):
        for h in (-0.3, 0.3, 1.0):
            assert abs(shin_from_horn(p, t, h, para)[0] - h) < 1e-6
    # hard stops: the nominal pose is allowed, the band around it is contiguous, and every
    # pose in it keeps the knee in range and both link angles outside the fold margin
    assert drive_pose_allowed(p, thigh, horn)
    stops = drive_stops(p)
    band = stops.band(thigh)
    assert band is not None and band[0] < horn < band[1]
    for k in range(21):
        h = band[0] + (band[1] - band[0]) * k / 20
        sh, hc, mu = shin_from_horn(p, thigh, h)
        assert KNEE_RANGE[0] - 1e-6 <= sh - thigh <= KNEE_RANGE[1] + 1e-6
        assert FOLD_MARGIN - 1e-6 <= hc <= math.pi - FOLD_MARGIN + 1e-6
        assert FOLD_MARGIN - 1e-6 <= mu <= math.pi - FOLD_MARGIN + 1e-6
    lo, hi = stops.thigh_range()
    assert lo <= thigh <= hi


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


def test_slalom_cones():
    import mujoco
    from evol_mech.skater.rollout import slalom_cones, CONE_LAYERS
    task = {"name": "slalom", "terrain": "flat", "objective": "slalom", "speed": 0.8, "duration": 10.0,
            "amplitude": 0.5, "wavelength": 3.0}
    xs = slalom_cones(task)
    # one cone per half wavelength at the path's lateral extremes, covering the course
    assert xs[0] == 0.75 and xs[1] == 2.25 and xs[-1] < 0.8 * 10 + 3.0 and len(xs) >= 7
    model, _ = make_model(DEFAULT_NORM, "flat", np.random.default_rng(0), task)
    cone_geoms = [g for g in range(model.ngeom)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("cone_")]
    assert len(cone_geoms) == len(xs) * CONE_LAYERS
    assert all(model.geom_contype[g] and model.geom_conaffinity[g] for g in cone_geoms)   # collidable
    # cones follow the ground on a heightfield
    model_r, _ = make_model(DEFAULT_NORM, "cobblestone", np.random.default_rng(0), task)
    assert model_r.ngeom == model.ngeom
    # no cones on the other objectives; the metric is reported on the slalom
    model_s, _ = make_model(DEFAULT_NORM, "flat", np.random.default_rng(0), {**task, "objective": "skate"})
    assert model_s.ngeom == model.ngeom - len(cone_geoms)
    r = rollout(DEFAULT_NORM, {**task, "duration": 1.0}, 0)
    assert r["cones"] == len(slalom_cones({**task, "duration": 1.0})) == 3 and r["cones_hit"] == 0


def test_legged_cycles():
    import mujoco
    from evol_mech.skater.controller import SkaterController
    p = decode(DEFAULT_NORM)
    model, _ = make_model(DEFAULT_NORM, "flat", np.random.default_rng(0))
    walk = SkaterController(model, p, {"objective": "walk", "speed": 0.4, "terrain": "flat"})
    T = 1 / p["step_hz"]
    # the left foot steps first: at t = 0 it is at its swing start (behind, on the floor) and the
    # right foot is in stance ahead of it; legs half a cycle apart; the stance foot travels back,
    # the swing foot comes forward lifted
    WALK_DUTY = p["step_duty"]
    s0 = walk.gait_offsets(0.0)
    assert abs(s0["l"][0] + p["step_len"] / 2) < 1e-9 and s0["l"][1] == 0.0 and s0["r"][0] > 0.0
    mid_swing = T * (1 - WALK_DUTY) / 2
    assert abs(walk.gait_offsets(mid_swing)["l"][1] - p["step_lift"]) < 1e-9
    t_td = T * (1 - WALK_DUTY)                                              # the left foot's touchdown, ahead
    l0 = walk.gait_offsets(t_td)["l"]
    assert abs(l0[0] - p["step_len"] / 2) < 1e-9 and abs(l0[1]) < 1e-9
    assert walk.gait_offsets(t_td + T * WALK_DUTY / 2)["l"][0] < l0[0]
    r_td = walk.gait_offsets(t_td + T / 2)["r"]
    assert abs(r_td[0] - l0[0]) < 1e-9 and abs(r_td[1] - l0[1]) < 1e-9
    hop = SkaterController(model, p, {"objective": "hop", "terrain": "flat"})
    bj = SkaterController(model, p, {"objective": "broad_jump", "speed": 0.5, "terrain": "flat"})
    Tj = 1 / p["jump_hz"]
    assert hop.gait_offsets(0.4 * Tj)["l"][1] == -p["jump_crouch"]          # crouched before the thrust
    assert hop.gait_offsets(0.5 * Tj)["l"] == (0.0, p["jump_ext"], False)   # hop: straight up
    gx, gz, fb = bj.gait_offsets(0.5 * Tj)["l"]                             # broad jump: feet back
    assert abs(gx + p["jump_lean"]) < 1e-9 and gz == p["jump_ext"] and not fb
    assert hop.legged and not SkaterController(model, p, {"objective": "skate", "terrain": "flat"}).legged
    # swing-side fan assist: mid-swing of the left foot the left fan runs harder than the right
    data = make_data(model, p)
    walk.act(data, 0.0)
    thr0 = walk.last_throttle
    walk.act(data, mid_swing)
    thr1 = walk.last_throttle
    assert abs(thr1[0] - thr0[0] - p["step_fan"]) < 1e-6 and abs(thr1[1] - thr0[1]) < 1e-6


def test_hip_roll_places_the_feet():
    import mujoco
    from evol_mech.skater.controller import SkaterController
    from evol_mech.skater.structure import HIP_ROLL_RANGE
    p = decode(DEFAULT_NORM)
    model, _ = make_model(DEFAULT_NORM, "flat", np.random.default_rng(0))
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    # roll replaces yaw at the hip; the ankle is sprung in pitch and roll
    assert jid("hip_yaw_l") < 0 and jid("hip_roll_l") >= 0 and jid("ankle_roll_r") >= 0
    assert np.allclose(model.jnt_axis[jid("hip_roll_l")], [1, 0, 0])
    assert abs(model.jnt_range[jid("hip_roll_l")][1] - HIP_ROLL_RANGE) < 1e-4
    assert model.jnt_stiffness[jid("ankle_roll_l")] > 0 and aid("hip_roll_r") >= 0
    # a rolled body (positive = right side down) shifts both feet to the right: negative roll commands
    data = make_data(model, p)
    data.qpos[3:7] = [math.cos(0.1), math.sin(0.1), 0.0, 0.0]
    stand = SkaterController(model, p, {"objective": "stand", "terrain": "flat"})
    c = stand.act(data, 0.0)
    assert c[aid("hip_roll_l")] < 0 and c[aid("hip_roll_r")] < 0
    # level body, wider stance: left foot out (+), right foot out (-)
    data = make_data(model, p)
    wide = SkaterController(model, {**p, "stance_w": 0.04}, {"objective": "stand", "terrain": "flat"})
    c = wide.act(data, 0.0)
    assert c[aid("hip_roll_l")] > 0 and c[aid("hip_roll_r")] < 0
    # walk: mid-swing of the left foot the body sits over the right foot (shifted right),
    # so both legs roll the feet to the left; the opposite half a cycle later
    walk = SkaterController(model, p, {"objective": "walk", "speed": 0.4, "terrain": "flat"})
    T = 1 / p["step_hz"]
    WALK_DUTY = p["step_duty"]
    mid_swing = T * (1 - WALK_DUTY) / 2                 # the cycle opens with the left foot's swing
    amp = p["walk_shift"] * p["hip_width"] / 2
    assert walk.body_shift(mid_swing) < -0.9 * amp
    assert walk.body_shift(mid_swing + T / 2) > 0.9 * amp
    c = walk.act(make_data(model, p), mid_swing)
    assert c[aid("hip_roll_l")] > 0 and c[aid("hip_roll_r")] > 0
    # the default stance is unchanged: zero roll, feet under the hips
    assert SkaterController(model, p, {"objective": "skate", "speed": 0.6, "terrain": "flat"}).body_shift(0.3) == 0.0


def test_walker_archetype_walks_and_walk_fitness_counts_the_feet():
    from evol_mech.skater.spec import encode, N_PARAMS
    from evol_mech.skater.rollout import FALL_COST
    assert N_PARAMS == 77 and FALL_COST == {"walk": 0.3, "run": 0.1, "course": 0.1}
    phys = json.load(open("configs/archetypes/skater_walker_evolved6.json"))["phys"]     # levelled sole, 2.5 Hz cap
    task = {"name": "walk_flat", "terrain": "flat", "objective": "walk", "speed": 0.4, "duration": 8.0, "step_hz_max": 2.5}
    kw = dict(motor_energy_cost=0.0005)                 # as configs/skater_walk*.json price it
    r = rollout(encode(phys), task, 0, **kw)
    # the evolved gait walks: forward progress of the feet, no fall, fans near idle, feet picked up
    assert not r["fell"] and r["feet_x_m"] > 2.5 and abs(r["feet_x_m"] - r["distance_x"]) < 0.3
    assert r["fan_energy"] < 0.3 and 0.1 < r["stride_m"] < 0.25 and r["swing_clear_m"] > 0.02
    # stepping outscores standing still on the walk task by a clear margin, through the grace
    # factor (the walk is judged on stride x grace, not speed): a stander's gait score is nil
    assert 0.25 < r["grace"] < 0.5 and r["fan_duty"] < 0.1 and r["sway_rad"] < 0.15 and r["bob_m"] < 0.01
    r_still = rollout(encode({**phys, "step_len": 0.02, "walk_shift": 0.0}), task, 0, **kw)
    assert r_still["feet_x_m"] < 0.3 and r["fitness"] > r_still["fitness"] + 0.15
    # a walk slower than the speed gate earns a share of its gait score: the gate is what it says
    from evol_mech.skater.rollout import WALK_SPEED_GATE
    r_g = rollout(encode(phys), {**task, "speed_gate": 10.0, "clear_weight": 0.0, "fan_duty_cost": 0.0}, 0, **kw)
    r_f = rollout(encode(phys), {**task, "clear_weight": 0.0, "fan_duty_cost": 0.0}, 0, **kw)
    assert r["feet_x_m"] / 8.0 > WALK_SPEED_GATE and r_g["fitness"] < r_f["fitness"]
    # the same walk with the fans off scores higher: no fan duty, no fan energy, quieter pitch servos
    r_off = rollout(encode(phys), task, 0, fans_enabled=False, **kw)
    assert not r_off["fell"] and r_off["fan_duty"] == 0.0 and r_off["fitness"] > r["fitness"]
    # a smoother body is worth more of the same stride: the grace factor is 1 / (1 + sum of ratios)
    from evol_mech.skater.rollout import _grace, SWAY_REF
    q = {"sway_rad": SWAY_REF, "bob_m": 0.0, "surge_ms": 0.0, "weave_m": 0.0}
    assert abs(_grace(task, q, 0.0) - 0.5) < 1e-9 and _grace(task, {**q, "sway_rad": 0.0}, 0.0) == 1.0
    assert abs(_grace({**task, "sway_ref": 2 * SWAY_REF}, q, 0.0) - 2 / 3) < 1e-9
    # ... and the track beyond the hip width is charged the same way (feet under the hips are free)
    from evol_mech.skater.rollout import TRACK_REF
    assert _grace(task, {**q, "track_m": 0.14}, 0.0, 0.14) == 0.5 and abs(_grace(task, {**q, "track_m": 0.14 + TRACK_REF}, 0.0, 0.14) - 1 / 3) < 1e-9
    assert phys["hip_width"] - 0.01 < r["track_m"] < phys["hip_width"] + 2 * phys["stance_w"] + 0.03   # the measured track is the feet's separation
    r_stand = rollout(encode(phys), {"name": "stand", "terrain": "flat", "objective": "stand", "duration": 1.0}, 0)
    assert not r_stand["fell"] and r_stand["time_upright"] >= 0.9

def test_walk_task_can_cap_the_cadence():
    from evol_mech.skater.spec import encode
    phys = json.load(open("configs/archetypes/skater_walker_evolved6.json"))["phys"]   # evolved under a 2.5 Hz cap
    task = {"name": "walk_flat", "terrain": "flat", "objective": "walk", "speed": 0.4, "duration": 2.0, "step_hz_max": 2.0}
    r = rollout(encode(phys), task, 0, motor_energy_cost=0.0005)
    # stride_m divides the feet progress by the capped step count, so the cap shows there
    assert abs(r["stride_m"] - r["feet_x_m"] / (2.0 * r["time_upright"])) < 1e-6


def test_static_legged_stance_holds_upright_and_crouched():
    """The levelled sole and stance_x: a legged stance with both feet planted holds at 0 and 3 cm
    of crouch (before, a 3 cm crouch tipped the sole onto an edge and the body over). A 6 cm
    crouch is marginal on these gains (it held only with the walk sway's old phase helping)."""
    import mujoco
    from evol_mech.skater.spec import encode
    from evol_mech.skater.rollout import make_model, make_data, SUBSTEPS, _fallen
    from evol_mech.skater.controller import SkaterController, CONTROL_DT
    from evol_mech.sim.controller import quat_to_roll_pitch
    phys = dict(json.load(open("configs/archetypes/skater_walker_evolved5.json"))["phys"],
                kp_foot_leg=0.0, kd_foot_leg=0.0, ki_foot_leg=0.0, kv_foot_x=0.1, stance_x=0.01,
                run_kp_leg=0.0, run_kv_x=0.1, run_stance_x=0.01, walk_shift=0.0)      # a static stance: no weight shift
    task = {"name": "run", "objective": "run", "terrain": "flat", "duration": 3.0, "speed": 0.0}
    for crouch in (0.0, 0.03):
        model, p = make_model(encode(phys), "flat", np.random.default_rng(0), task)
        data = make_data(model, p)
        ctrl = SkaterController(model, p, task, True)
        ctrl.gait_offsets = lambda t, c=crouch: {s: (0.0, -c * min(1.0, t / 0.5), False) for s in "lr"}
        for k in range(300):
            data.ctrl[:] = ctrl.act(data, k * CONTROL_DT)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            roll, pitch = quat_to_roll_pitch(data.qpos[3:7])
            assert not _fallen(float(data.qpos[2]), roll, pitch, 0.345), f"fell at {k * CONTROL_DT:.2f} s with {crouch} m crouch"
        assert abs(pitch) < 0.2 and abs(data.qpos[2] - (0.345 - crouch)) < 0.03

def test_runner_archetype_bounds_with_real_flights_and_the_run_fitness_pays_for_them():
    from evol_mech.skater.spec import encode
    phys = json.load(open("configs/archetypes/skater_runner_evolved.json"))["phys"]
    task = {"name": "run_flat", "terrain": "flat", "objective": "run", "speed": 1.5, "duration": 6.0}
    r = rollout(encode(phys), task, 0, motor_energy_cost=0.0005)
    # a bound: many real flights (hip rising >= 1 cm), a good share of the time in the air, feet picked up
    assert not r["fell"] and r["feet_x_m"] > 1.0 and r["n_flights"] >= 8 and r["flight_frac"] > 0.25
    assert r["swing_clear_m"] > 0.03 and 0.1 < r["stride_m"] < 0.35
    # the flight-length, clearance and bob terms are what they say (the flight share gates, it does not pay)
    from evol_mech.skater.rollout import FLIGHT_GATE, FLIGHT_LEN_CAP
    r0 = rollout(encode(phys), {**task, "flight_len_weight": 0.0, "clear_weight": 0.0, "bob_cost": 0.0}, 0, motor_energy_cost=0.0005)
    gate = min(1.0, r["flight_frac"] / FLIGHT_GATE)
    expect = gate * 1.0 * min(r["flight_len_m"], FLIGHT_LEN_CAP) + 2.0 * min(r["swing_clear_m"], 0.05) - 2.0 * r["bob_m"]
    assert gate == 1.0 and abs((r["fitness"] - r0["fitness"]) - expect) < 1e-6
    # a hop in place earns nothing: the gate and the speed term are both multiplicative on progress
    r_fw = rollout(encode(phys), {**task, "flight_gate": 10.0}, 0, motor_energy_cost=0.0005)   # 10 x the flight share: gated down
    assert r_fw["fitness"] < r["fitness"]

def test_all_objectives_run_and_default_stands():
    for obj, extra in (("skate", {"speed": 0.6}), ("crab", {"speed": 0.3}), ("slalom", {"speed": 0.6}), ("stand", {}),
                       ("walk", {"speed": 0.4}), ("run", {"speed": 1.5}), ("hop", {}), ("broad_jump", {"speed": 0.5})):
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


def test_course_blends_one_cycle_between_the_gaits_and_scores_each_segment():
    import mujoco
    from evol_mech.skater.spec import encode
    from evol_mech.skater.controller import SkaterController
    p = decode(DEFAULT_NORM)
    model, _ = make_model(DEFAULT_NORM, "flat", np.random.default_rng(0))
    # the walk is the stepping cycle with no crouch; the run is the same cycle crouched with a short duty
    walk = SkaterController(model, p, {"objective": "walk", "speed": 0.4, "terrain": "flat"})
    assert walk._walk_leg(0.3) == walk._leg(0.3, p["step_len"], p["step_duty"], p["step_lift"], 0.0, 0.0)
    assert walk._run_leg(0.3, 1.0) == walk._leg(0.3, p["run_len"], p["run_duty"], p["run_lift"], p["run_crouch"], p["run_ext"])
    profile = [["stand", 1.0], ["walk", 3.0], ["run", 3.0], ["stand", 4.0]]
    c = SkaterController(model, p, {"objective": "course", "speed": 0.4, "terrain": "flat", "profile": profile})
    data = make_data(model, p)
    state = {}
    for k in range(1100):
        t = k * 0.01
        c.act(data, t)
        state[round(t, 2)] = (c.amp, c.blend, c.phase, c.gait_offsets(t))
    assert state[0.5][0] == 0.0 and state[0.5][3]["l"] == (0.0, 0.0, True)      # standing: no step, feet parked
    assert abs(state[2.5][0] - 1.0) < 1e-9 and state[2.5][1] == 0.0             # walking after trans_s: full amplitude, no run in the mix
    assert 0.3 < state[4.5][1] < 0.7 < state[5.5][1] <= 1.0                        # the mix slews over trans_s into the run
    assert c.course_gait(4.5) == "run" and c.course_gait(8.0) == "stand" and c.course_gait(99.0) == "stand"
    assert state[10.95][0] == 0.0 and state[10.95][2] == p["step_duty"]           # stopped again: amplitude gone, phase parked at the left swing start
    # every start is a left step: at the first stepping instant the left foot is at its swing start (behind, lifting)
    first = next(t for t in sorted(state) if state[t][0] > 0.0)
    assert state[first][3]["l"][0] < 0.0 < state[first][3]["r"][0]
    # and the stop is a closing step: the right foot lands beside the left before the amplitude fades
    # (the phase frozen at 0.5, the right foot's landing, while the amplitude is still up)
    closes = [t for t in sorted(state) if 7.0 <= t <= 11.0 and abs(state[t][3]["r"][0] - state[t][3]["l"][0]) < 1e-9
              and state[t][0] > 0.5 and state[t][2] == 0.5]
    assert closes and state[round(closes[0] - 0.01, 2)][0] > 0.5 and c.closing is None
    opens_running = SkaterController(model, p, {"objective": "course", "terrain": "flat", "profile": [["run", 2.0]]})
    assert opens_running.blend == 1.0
    # the seed completes the course: both walks, a bounding run, four changes survived, and the transition
    # bonus is what it says
    phys = json.load(open("configs/archetypes/skater_courser_evolved3.json"))["phys"]     # evolved under the left-first start and the braking stop
    task = {"name": "course", "terrain": "flat", "objective": "course", "speed": 0.4, "step_hz_max": 2.5}
    r = rollout(encode(phys), task, 0, motor_energy_cost=0.0005)
    assert not r["fell"] and r["course_done"] == 1.0 and r["time_upright"] == 36.0
    # the walks move (the speed gate), the run flies (the flight gate), the feet are picked up
    assert r["seg1_walk.speed"] > 0.1 and r["seg3_walk.speed"] > 0.1 and r["seg2_run.flight_frac"] > 0.15 and r["seg2_run.swing_clear_m"] > 0.03
    assert r["seg2_run.speed"] > r["seg1_walk.speed"] and 0.2 < r["seg1_walk.grace"] < 1.0
    assert all(r[f"trans{k}_{g}.survived"] == 1.0 and r[f"trans{k}_{g}.pitch_max_deg"] < 30 for k, g in ((1, "walk"), (2, "run"), (3, "walk"), (4, "stand")))
    assert abs(r["seg4_stand.progress_m"]) < 0.15                   # the stop: the feet settle within a few cm
    r0 = rollout(encode(phys), {**task, "trans_weight": 0.0}, 0, motor_energy_cost=0.0005)
    expect = sum(r[f"trans{k}_{g}.score"] for k, g in ((1, "walk"), (2, "run"), (3, "walk"), (4, "stand")))
    assert expect > 0.5 and abs((r["fitness"] - r0["fitness"]) - expect) < 1e-6


def test_repeats_average_tilted_starts():
    from evol_mech.skater.rollout import evaluate
    task = {"name": "stand", "terrain": "flat", "objective": "stand", "duration": 1.0, "pose_noise_deg": 3.0}
    a, b = (rollout(DEFAULT_NORM, task, sd) for sd in (0, 100))
    assert a["fitness"] != b["fitness"]                      # the tilt differs with the seed
    assert rollout(DEFAULT_NORM, task, 0)["fitness"] == a["fitness"]   # and is reproducible
    e = evaluate(DEFAULT_NORM, {"tasks": [{**task, "repeats": 2}]}, 0)
    assert abs(e["fitness"] - (a["fitness"] + b["fitness"]) / 2) < 1e-9
    assert e["stand.fitness_min"] == min(a["fitness"], b["fitness"])
