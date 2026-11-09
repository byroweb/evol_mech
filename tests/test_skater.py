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


def test_pantograph_kinematics_closed_chain_and_lever():
    from evol_mech.skater.structure import (is_pantograph, panto_points, panto_ik, panto_pose_allowed, panto_reach,
                                            panto_nominal, nominal_foot, leg_reach, hip_roll_range, horn_stiffness,
                                            series_kp, HORN_RANGE, check_buildable)
    from evol_mech.skater.mjcf import SERVO_KV_FRAC
    from evol_mech.body.hardware import servo_from_params
    p0 = decode(DEFAULT_NORM)
    assert not is_pantograph(p0) and leg_reach(p0) == p0["thigh_len"] + p0["shin_len"]
    # the leg lengths of the 25 mm horn seed (a horn that short reaches a step only with
    # short rods and long extensions, so the spec's defaults are not a buildable set with it)
    seed = json.load(open("configs/archetypes/skater_walker_panto5_seed.json"))["phys"]
    leg = {k: seed[k] for k in ("horn_len", "rod_len", "rod_ext", "strut_len", "servo_dx", "crouch")}
    p = decode(encode({**p0, **leg, "leg_drive": 1.0, "hip_width": 0.10}))
    assert is_pantograph(p) and p["horn_len"] == pytest.approx(0.025)
    # the parked crouch: ankle straight below the hip point at reach x cos(crouch / 2)
    hf, hr = panto_nominal(p)
    k = panto_points(p, hf, hr)
    assert abs(k["ankle"][0]) < 1e-6 and abs(-k["ankle"][1] - panto_reach(p) * math.cos(p["crouch"] / 2)) < 1e-4
    assert nominal_foot(p) == pytest.approx(k["ankle"]) and 0 < panto_reach(p) <= p["horn_len"] + p["rod_len"] + 2 * p["rod_ext"] + p["strut_len"]
    # the pin lies on both rods, the rods cross there and the ankle closes the parallelogram
    t1, t2, pin, u1, u2 = k["t1"], k["t2"], k["pin"], k["u1"], k["u2"]
    assert math.hypot(pin[0] - t1[0], pin[1] - t1[1]) == pytest.approx(p["rod_len"])
    assert math.hypot(pin[0] - t2[0], pin[1] - t2[1]) == pytest.approx(p["rod_len"])
    assert pin[1] < min(t1[1], t2[1]) and u1[0] < 0 < u2[0]          # below the horn tips; front rod runs back, rear rod forward
    e, st = p["rod_ext"], p["strut_len"]
    assert k["ankle"] == pytest.approx((pin[0] + e * u1[0] + (e + st) * u2[0], pin[1] + e * u1[1] + (e + st) * u2[1]))
    # IK round trip over a step's workspace, warm-started, every pose clear of the folds
    for dx, dz in ((0.0, 0.0), (0.05, 0.0), (-0.05, 0.0), (0.0, 0.04), (0.03, 0.02)):
        a, b, res = panto_ik(p, k["ankle"][0] + dx, k["ankle"][1] + dz, (hf, hr))
        assert res < 1e-5 and panto_pose_allowed(p, a, b)
        assert panto_points(p, a, b)["ankle"] == pytest.approx((k["ankle"][0] + dx, k["ankle"][1] + dz), abs=1e-5)
    # out of reach: the nearest allowed pose, still finite and allowed
    a, b, res = panto_ik(p, 0.0, -1.0, (hf, hr))
    assert math.isfinite(res) and res > 0.1 and panto_pose_allowed(p, a, b)
    assert not panto_pose_allowed(p, HORN_RANGE[1] + 0.1, hr)
    check_buildable(p)
    with pytest.raises(InvalidBody):                       # rods too short to meet across the servo spacing
        check_buildable({**p, "rod_len": 0.08, "servo_dx": 0.10, "horn_len": 0.01, "rod_ext": 0.03})
    # the lever and the flexing horns reflect onto the hinge actuators. Pantograph body: the
    # pivots are at the hip centre and the bracket is the arm, so the ratio is derived and the
    # roll_ratio gene is ignored; four-bar body: the gene
    from evol_mech.skater.structure import roll_arm, roll_ratio, ROLL_PIVOT_Y
    assert roll_arm(p0) == 0.0 and roll_ratio({**p0, "roll_ratio": 5.0}) == 5.0 and hip_roll_range({**p0, "roll_ratio": 5.0}) == pytest.approx(math.radians(26))
    assert roll_arm(p) == pytest.approx(0.05 - ROLL_PIVOT_Y) and roll_ratio(p) == pytest.approx((0.05 - ROLL_PIVOT_Y) / p["horn_len"])
    assert roll_ratio({**p, "roll_ratio": 5.0}) == roll_ratio(p) and hip_roll_range(p) == pytest.approx(math.radians(30))
    # a wide hip: the roll lever's own toggle caps the roll, not the horn's sweep. The horn is
    # horizontal with a vertical link to the bracket end, so the end can rise or drop by about a
    # horn length: asin(horn / arm) to first order. The round-9 body (144 mm arm, 25 mm horn)
    # toggles at 8.6 deg in the simulation, where the old sweep / ratio formula said 22
    from evol_mech.skater.structure import roll_lever_range
    wide = {**p, "hip_width": 0.32, "horn_len": 0.02}                      # arm 0.148 / horn 0.02 = 7.4
    assert hip_roll_range(wide) == pytest.approx(roll_lever_range(wide))
    assert roll_lever_range(wide) < math.radians(130) / 7.4
    r9 = {**p, "hip_width": 0.311, "horn_len": 0.025}
    assert math.radians(8.3) < roll_lever_range(r9) < math.radians(8.8)
    assert horn_stiffness(p) == pytest.approx(1000.0) and series_kp(50.0, 50.0) == pytest.approx(25.0)
    pl = {**p, "horn_k_log10": 1.0}
    model, pm = make_model(encode(pl), "flat", np.random.default_rng(0))
    servo = servo_from_params(pm)
    act = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    kp_leg = series_kp(servo.kp, 10.0)
    kp_roll = kp_leg * roll_ratio(pm) ** 2
    assert model.actuator_gainprm[act("hip_pitch_l"), 0] == pytest.approx(kp_leg, rel=1e-3)
    assert model.actuator_biasprm[act("knee_l"), 2] == pytest.approx(-kp_leg * SERVO_KV_FRAC, rel=1e-2)
    assert model.actuator_gainprm[act("hip_roll_l"), 0] == pytest.approx(kp_roll, rel=1e-3)
    assert model.actuator_forcerange[act("hip_roll_l"), 1] == pytest.approx(roll_ratio(pm) * servo.torque_nm, rel=1e-3)
    # built as a real closed chain: the loops close at the crouch, the sim's ankle is the kinematics' ankle
    model, pm = make_model(encode(p), "flat", np.random.default_rng(0))
    assert model.neq >= 4 and all(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) in
                                  {"pin_l", "pin_r", "node_l", "node_r", "roll_l", "roll_r"} for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT)
    data = make_data(model, pm)
    mujoco.mj_forward(model, data)
    body = lambda n: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)]
    rel = body("foot_l") - body("hip")
    assert rel[0] == pytest.approx(k["ankle"][0], abs=1e-4) and rel[2] == pytest.approx(k["ankle"][1], abs=1e-4)
    assert rel[1] == pytest.approx(0.05, abs=1e-4)                        # the leg hangs at the leg plane, the pivot at the centre
    assert body("hip_roll_l")[1] - body("hip")[1] == pytest.approx(ROLL_PIVOT_Y, abs=1e-6)
    assert np.abs(data.efc_pos[:model.neq * 3]).max() < 1e-6 if data.nefc else True
    # ... and it stands on it, the loops holding, with the same actuator set as the four-bar leg
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)}
    assert names == set(ACTUATOR_NAMES)
    r = rollout(encode(p), {"name": "stand", "terrain": "flat", "objective": "stand", "duration": 3.0}, 0)
    assert r["time_upright"] == 3.0 and r["pitch_rms"] < 0.1
    # a walk task charges standing still like a fall at the start: the pantograph's first round was won by a stander
    from evol_mech.skater.rollout import STAND_COST
    stander = {**seed, "step_len": 0.0, "step_lift": 0.0}                # the seed parked
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "duration": 3.0, "step_hz_max": 2.5}
    r1, r0 = rollout(encode(stander), task, 0), rollout(encode(stander), {**task, "stand_cost": 0.0}, 0)
    gate = min(1.0, (r1["feet_x_m"] / 3.0) / 0.15)
    assert r1["time_upright"] == 3.0 and abs((r0["fitness"] - r1["fitness"]) - STAND_COST * (1 - gate)) < 1e-6
    # the 25 mm horn seed walks forward on it (the older 6 cm horn archetypes are history
    # under the 25 mm cap: encode clamps their horn and the leg no longer reaches a step)
    r = rollout(encode(seed), {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "duration": 6.0,
                                 "step_hz_max": 2.5}, 0)
    assert r["time_upright"] == 6.0 and r["feet_x_m"] > 0.5


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
    assert N_PARAMS == 88 and FALL_COST == {"walk": 0.3, "run": 0.1, "course": 0.1}
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


def test_foot_sensors_read_the_stance_and_survive_dropouts():
    """Phase 0 of the test plan: two FSRs per foot. Standing, each foot carries about half the
    weight with contact on and the centre of pressure on the sole; a lifted foot reads nothing;
    a channel that always drops reads zero and never reports contact."""
    from evol_mech.skater.sensors import FootSensors
    from evol_mech.skater.controller import SkaterController, CONTROL_DT
    from evol_mech.sim.rollout import SUBSTEPS
    seed = json.load(open("configs/archetypes/skater_walker_panto5_seed.json"))["phys"]
    stander = decode(encode({**seed, "step_len": 0.0, "step_lift": 0.0}))
    task = {"name": "stand", "terrain": "flat", "objective": "stand", "duration": 1.0}
    r = rollout(encode(stander), task, 0)
    assert r["time_upright"] >= 1.0 and r["sensor_stance_frac"] > 0.9
    assert 0.3 < r["touchdown_load_bw"] < 1.2            # the stand's first contact rise, both feet
    assert r["cop_excursion_m"] < stander["sole_len"]
    # a direct read: build, settle, inspect
    rng = np.random.default_rng(0)
    model, p = make_model(encode(stander), "flat", rng, task)
    data = make_data(model, p)
    ctrl = SkaterController(model, p, task)
    sens = ctrl.sensors
    for k in range(60):
        data.ctrl[:] = ctrl.act(data, k * CONTROL_DT)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
    for s in "lr":
        assert sens.contact[s] and 0.25 < sens.load[s] < 0.8, (s, sens.load[s])
        assert abs(sens.cop[s]) < p["sole_len"] / 2
    assert sens.servo_load.max() < 0.6                    # no servo near stall standing
    # a sensor that drops every reading
    dead = FootSensors(model, p, ctrl.mass, CONTROL_DT, rng=np.random.default_rng(1), dropout=1.0)
    for _ in range(5):
        dead.update(data)
    assert all(dead.load[s] == 0.0 and not dead.contact[s] for s in "lr") and dead.dropped == 20


def test_real_servo_model_delays_holds_and_slows_the_commands():
    """Phase 0 of the test plan: with task servo_model "real" the six leg servos receive the
    command issued 20 ms ago, refreshed at 50 Hz, moved only past the pulse deadband, and slew
    slower under load; the fans pass straight through; the default stays the ideal servo."""
    from evol_mech.skater.controller import SkaterController, CONTROL_DT
    from evol_mech.body.hardware import SERVO_LATENCY_S, SERVO_CMD_DEADBAND_RAD
    seed = json.load(open("configs/archetypes/skater_walker_panto5_seed.json"))["phys"]
    stander = decode(encode({**seed, "step_len": 0.0, "step_lift": 0.0}))
    rng = np.random.default_rng(0)
    model, p = make_model(encode(stander), "flat", rng, {"objective": "stand", "terrain": "flat"})
    ideal = SkaterController(model, p, {"objective": "stand", "terrain": "flat"})
    real = SkaterController(model, p, {"objective": "stand", "terrain": "flat", "servo_model": "real"})
    assert not ideal.real_servo and real.real_servo
    # the link alone: a step in the command reaches the servos after the latency, not before
    nu = model.nu
    for k in range(6):
        c = np.zeros(nu)
        c[real.servo_acts] = 1.0 if k >= 1 else 0.0          # the command steps at k = 1
        c[real.idx["fan_thrust_l"]] = float(k)
        out = real._servo_link(c)
        assert out[real.idx["fan_thrust_l"]] == float(k)      # fans untouched
        if k < 1 + round(SERVO_LATENCY_S / CONTROL_DT):
            assert np.all(out[real.servo_acts] == 0.0), (k, out[real.servo_acts])
    assert np.all(out[real.servo_acts] == 1.0)
    # a change inside the deadband never moves the held command
    c[real.servo_acts] = 1.0 + 0.5 * SERVO_CMD_DEADBAND_RAD
    for _ in range(6):
        out = real._servo_link(c)
    assert np.all(out[real.servo_acts] == 1.0)
    # under load the slew step shrinks: a servo at stall creeps
    data = make_data(model, p)
    real._data = data
    data.actuator_force[real.idx["knee_l"]] = model.actuator_forcerange[real.idx["knee_l"]][1]
    real.servo_cmd["knee_l"] = 0.0
    slow = real._slew("knee_l", 1.0)
    data.actuator_force[real.idx["knee_l"]] = 0.0
    real.servo_cmd["knee_l"] = 0.0
    fast = real._slew("knee_l", 1.0)
    assert 0.0 < slow < 0.2 * fast
    # the stand runs under both models and the default result is the ideal one
    task = {"name": "s", "terrain": "flat", "objective": "stand", "duration": 1.0}
    r0 = rollout(encode(stander), task, 0)
    r1 = rollout(encode(stander), {**task, "servo_model": "ideal"}, 0)
    r2 = rollout(encode(stander), {**task, "servo_model": "real"}, 0)
    assert r0["fitness"] == r1["fitness"] and r2["fitness"] != r1["fitness"] and r2["time_upright"] > 0.5


def test_result_reports_fall_direction_servo_peaks_and_foot_track():
    """Phase 0 of the test plan: the result says which way the body fell, how hard each servo
    worked per gait cycle (fraction of stall / no-load speed) and how far the ankles ran from
    the IK's target in the hip frame."""
    from evol_mech.skater.rollout import _fall_dir
    assert _fall_dir(False, 1.0, 1.0) == "none"
    assert _fall_dir(True, 0.1, 1.1) == "forward" and _fall_dir(True, 0.1, -1.1) == "back"
    assert _fall_dir(True, 1.1, 0.2) == "right" and _fall_dir(True, -1.1, 0.2) == "left"
    assert _fall_dir(True, 0.05, 0.05) == "drop"
    seed = json.load(open("configs/archetypes/skater_walker_panto5_seed.json"))["phys"]
    stander = decode(encode({**seed, "step_len": 0.0, "step_lift": 0.0}))
    r = rollout(encode(stander), {"name": "stand", "terrain": "flat", "objective": "stand", "duration": 1.5}, 0)
    assert r["time_upright"] >= 1.5 and r["fall_dir"] == "none"
    for j in ("hip_roll", "hip_pitch", "knee"):
        assert 0.0 < r[f"peak_torque_{j}"] <= 1.0 and 0.0 <= r[f"peak_speed_{j}"] < 3.0
    assert r["servo_peak_torque_frac"] >= max(r[f"peak_torque_{j}"] for j in ("hip_roll", "hip_pitch", "knee"))
    assert r["foot_track_y_rms_m"] < 0.01 and r["foot_track_rms_m"] < 0.05
    # a walker that falls names a direction
    walker = json.load(open("configs/archetypes/skater_walker_panto7.json"))["phys"]
    r = rollout(encode(walker), {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5,
                                 "duration": 12.0, "pose_noise_deg": 2.0, "servo_model": "real"}, 0)
    assert r["fell"] and r["fall_dir"] in ("forward", "back", "left", "right", "drop")


def test_rolled_plane_ik_lands_the_ankle_on_its_target():
    """Phase 0 of the test plan: the hip roll plus the planar IK solved in the rolled leg plane
    puts the ankle where the controller asked, laterally to the millimetre on the four-bar
    standing (roll about x leaves a fore-aft step with no lateral component, so there is
    nothing for a rolled-plane rotation to fix); the wide-hip walker's inboard feet stay
    centimetres short because the planted soles will not slide, not because of the solve."""
    from evol_mech.skater.controller import SkaterController, CONTROL_DT
    from evol_mech.sim.rollout import SUBSTEPS

    def settle(name: str, steps: int = 300):
        p = decode(encode(json.load(open(f"configs/archetypes/skater_walker_{name}.json"))["phys"]))
        model, p = make_model(encode(p), "flat", np.random.default_rng(0))
        data = make_data(model, p)
        ctrl = SkaterController(model, p, {"objective": "stand", "terrain": "flat"})
        hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
        for k in range(steps):
            data.ctrl[:] = ctrl.act(data, k * CONTROL_DT)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
        R = data.xmat[hip].reshape(3, 3)
        return {s: R.T @ (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{s}")] - data.xpos[hip]) - ctrl.foot_target[s]
                for s in "lr"}

    err = settle("hip3")
    for s in "lr":
        assert abs(err[s][1]) < 0.002, (s, err[s])
        assert np.linalg.norm(err[s]) < 0.02, (s, err[s])
    err = settle("panto9")
    assert all(abs(err[s][1]) > 0.005 for s in "lr")             # the documented squeeze, reported not hidden
    assert err["l"][1] * err["r"][1] < 0                          # mirrored: each foot short toward its own side


def test_static_feasibility_check():
    """Phase 0 of the test plan: the no-simulation body check. The round-9 pantograph has a
    walking step's workspace and its servos well inside stall, but fails on the roll range:
    its inboard stance needs 11 deg of hip roll and the roll lever toggles at 8.5 (found by
    the phase-1 stall diagnosis; the roll servos sat on that toggle at full torque for every
    run of round 9). At the lever's range it passes. The four-bar leg fails on fore / aft
    reach at the crouch (the shin drive's fold and the knee's 10 deg stop); feet asked inboard
    past the soles, or a servo too small for the body, fail by name."""
    from evol_mech.skater.feasibility import static_check, feasible, FORE_MIN, LIFT_MIN, STATIC_TORQUE_MAX
    p9 = decode(encode(json.load(open("configs/archetypes/skater_walker_panto9.json"))["phys"]))
    r = static_check(p9)
    assert not r["ok"] and len(r["reasons"]) == 1 and "hip roll" in r["reasons"][0], r["reasons"]
    assert r["lift_m"] >= LIFT_MIN and r["fore_m"] >= FORE_MIN and r["leg_torque_frac"] < STATIC_TORQUE_MAX
    assert 8.0 < r["roll_range_deg"] < 9.0 and r["roll_need_deg"] > r["roll_range_deg"]
    assert r["sole_gap_m"] > 0.01 and r["stance_w_used"] < 0
    assert 0.0 < r["roll_torque_frac"] < 0.2                      # weight is not what held the roll servos at stall
    r = static_check({**p9, "stance_w": -0.03})                   # a stance the lever reaches
    assert r["ok"] and not r["reasons"], r["reasons"]
    h3 = decode(encode(json.load(open("configs/archetypes/skater_walker_hip3.json"))["phys"]))
    r = static_check(h3)
    assert not r["ok"] and all("reach" in why for why in r["reasons"]) and r["lift_m"] > LIFT_MIN
    r = static_check({**p9, "stance_w": -0.2})
    assert any("soles" in why for why in r["reasons"])
    r = static_check({**p9, "servo_kgcm": 10.0})
    assert any("leg servo" in why for why in r["reasons"]) and not feasible({**p9, "servo_kgcm": 10.0})
    assert "not buildable" in static_check({**p9, "hip_width": 0.03})["reasons"][0]


def test_contact_driven_walk_runs_the_phase_off_the_feet():
    """Phase 1 item 1: with task gait_phase "contact" the walk's shared phase waits at the start
    until the first stance foot carries the body, holds briefly for a foot that has not found
    the floor by its touchdown mark (the clock then takes over), and starts each stance sweep
    from the foot's measured landing point. The default ("clock") is untouched: no events."""
    from evol_mech.skater.controller import SkaterController, CONTROL_DT, START_RAMP_S, START_TIMEOUT_S
    from evol_mech.sim.rollout import SUBSTEPS
    walker = json.load(open("configs/archetypes/skater_walker_panto8.json"))["phys"]
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 6.0, "pose_noise_deg": 2.0}
    clock = rollout(encode(walker), task, 0)
    assert clock["contact_events"] == {"early": 0, "late": 0, "clock": 0}
    contact = rollout(encode(walker), {**task, "gait_phase": "contact"}, 0)
    ev = contact["contact_events"]
    assert contact["time_upright"] >= 6.0 and contact["feet_x_m"] > 0.5
    assert ev["late"] + ev["clock"] + ev["early"] >= 6                 # a touchdown per half cycle, each resolved one way
    # the start: the phase parks at the left swing start until the right foot reads the load (or the timeout)
    model, p = make_model(encode(walker), "flat", np.random.default_rng(0), {**task, "gait_phase": "contact"})
    data = make_data(model, p)
    ctrl = SkaterController(model, p, {**task, "gait_phase": "contact"})
    started_at = None
    for k in range(int(1.2 / CONTROL_DT)):
        t = k * CONTROL_DT
        data.ctrl[:] = ctrl.act(data, t)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        if started_at is None and ctrl._gait_started:
            started_at = t
            assert abs(ctrl.cycles - p["step_duty"]) < 1e-9
    assert started_at is not None and START_RAMP_S <= started_at <= START_TIMEOUT_S + CONTROL_DT
    # after the start the stance anchor is the measured foot relative to the balance target
    assert abs(ctrl._anchor["r"] - (ctrl.foot_meas["r"][0] - ctrl._x_bal["r"])) < 0.05


def test_walk_stride_floor_gates_short_strides():
    """Phase 1 item 5, first piece: task `stride_min` folds a stride gate into the walk's speed
    gate. The phase-1a shuffler (2.7 cm steps, 3 cm/s) is already fully gated by its speed, so the
    floor changes nothing for it: the real servo chose it because every strider fell, not because
    the score let it travel. A strider under the floor keeps its score; a floor above its stride
    takes the same fraction of the gait score that the speed gate would."""
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 6.0, "pose_noise_deg": 2.0}
    shuffler = json.load(open("configs/archetypes/skater_walker_phase1a.json"))["phys"]
    a = rollout(encode(shuffler), {**task, "servo_model": "real", "gait_phase": "contact"}, 0)
    b = rollout(encode(shuffler), {**task, "servo_model": "real", "gait_phase": "contact", "stride_min": 0.10}, 0)
    assert a["stride_m"] < 0.05 and a["feet_x_m"] / 6.0 < 0.15 and b["fitness"] == a["fitness"]
    strider = json.load(open("configs/archetypes/skater_walker_panto8.json"))["phys"]
    a = rollout(encode(strider), task, 0)
    b = rollout(encode(strider), {**task, "stride_min": 0.10}, 0)
    c = rollout(encode(strider), {**task, "stride_min": 0.60}, 0)
    assert a["stride_m"] > 0.10 and b["fitness"] == a["fitness"] and c["fitness"] < a["fitness"] - 0.05


def test_capture_point_lands_the_swing_foot_ahead_of_the_body_velocity():
    """Phase 1 item 2: with task foot_placement "capture" the swing foot's landing moves by the hip
    velocity x sqrt(h_com / g) (the linear inverted pendulum's capture point), weighted in over the
    swing; the planted feet keep the kp / kd shift and the kv velocity genes are ignored. The
    default ("gains") is untouched."""
    from evol_mech.skater.controller import SkaterController, CONTROL_DT
    from evol_mech.sim.rollout import SUBSTEPS
    walker = json.load(open("configs/archetypes/skater_walker_phase1a.json"))["phys"]
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 6.0, "pose_noise_deg": 2.0}
    base = rollout(encode(walker), task, 0)
    cap = rollout(encode(walker), {**task, "foot_placement": "capture"}, 0)
    assert base["time_upright"] >= 6.0 and cap["time_upright"] >= 6.0
    assert cap["fitness"] != base["fitness"]
    # the swing weight follows the phase, and only the swing foot sees the capture offset
    model, p = make_model(encode(walker), "flat", np.random.default_rng(0), {**task, "foot_placement": "capture"})
    data = make_data(model, p)
    ctrl = SkaterController(model, p, {**task, "foot_placement": "capture"})
    seen_swing = seen_stance = False
    for k in range(200):
        t = k * CONTROL_DT
        data.ctrl[:] = ctrl.act(data, t)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        u = ctrl.cycles % 1.0
        if u < p["step_duty"]:
            assert ctrl.swing_w["l"] == 0.0
            seen_stance = True
        else:
            assert 0.0 <= ctrl.swing_w["l"] <= 1.0 and ctrl.swing_w["r"] == 0.0
            seen_swing = True
    assert seen_swing and seen_stance


def test_servo_lead_stands_the_round9_strider_under_the_real_servo():
    """Phase 1 item 3a: latency compensation (task servo_lead). Under the real servo the round-9
    strider falls within 6 s from the seed-0 tilted start; commanding for when the command lands
    (gait 40 ms ahead, tilt led by its rate) it stands the 12 s. Default 0 leaves scores untouched."""
    from evol_mech.skater.controller import SkaterController
    from evol_mech.skater.rollout import rollout, make_model
    p = json.load(open("configs/archetypes/skater_walker_panto9.json"))["phys"]
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 12.0,
            "pose_noise_deg": 2.0, "servo_model": "real", "gait_phase": "clock"}
    plain = rollout(encode(p), task, 0)
    led = rollout(encode(p), {**task, "servo_lead": "auto"}, 0)
    assert plain["fell"] and plain["time_upright"] < 6.0, plain["time_upright"]
    assert not led["fell"] and led["time_upright"] == pytest.approx(12.0, abs=0.02) and led["feet_x_m"] > 1.0
    model, phys = make_model(encode(p), "flat", np.random.default_rng(0), task)
    assert SkaterController(model, phys, task, True).lead == 0.0
    assert SkaterController(model, phys, {**task, "servo_lead": "auto"}, True).lead == pytest.approx(0.04)


def test_workspace_clamp_keeps_the_pantograph_ik_on_its_branch_and_stands_the_round7_strider():
    """Phase 1 item 3b: the workspace clamp (task leg_reach 'clamp'). The round-7 strider ran its
    stance leg past the pantograph's reach every cycle and lived on the IK's nearest-pose fallback;
    under the real servo the fallback wandered onto the other solution branch and it fell in 1-4 s.
    Clamped to the reach (and led for the latency) it strides 12 s from the seed-0 tilted start;
    under the ideal servo the clamp changes the score by a hair and takes the slew saturation from
    0.19 to under 0.05, the chatter having been the fallback. Default 'free' keeps every score."""
    from evol_mech.skater.controller import SkaterController, REACH_MARGIN
    from evol_mech.skater.structure import panto_reach
    p = json.load(open("configs/archetypes/skater_walker_panto7.json"))["phys"]
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 12.0,
            "pose_noise_deg": 2.0, "servo_model": "real", "gait_phase": "clock", "servo_lead": "auto"}
    free = rollout(encode(p), task, 0)
    clamped = rollout(encode(p), {**task, "leg_reach": "clamp"}, 0)
    assert free["fell"] and free["time_upright"] < 5.0, free["time_upright"]
    assert not clamped["fell"] and clamped["time_upright"] == pytest.approx(12.0, abs=0.02)
    assert clamped["stride_m"] > 0.10 and clamped["feet_x_m"] > 1.2 and clamped["swing_clear_m"] > 0.02
    ideal = {**task, "servo_model": "ideal", "servo_lead": 0.0}
    a, b = rollout(encode(p), ideal, 0), rollout(encode(p), {**ideal, "leg_reach": "clamp"}, 0)
    assert a["servo_sat_frac"] > 0.1 and b["servo_sat_frac"] < 0.05 and abs(a["fitness"] - b["fitness"]) < 0.1
    model, phys = make_model(encode(p), "flat", np.random.default_rng(0), task)
    c = SkaterController(model, phys, {**task, "leg_reach": "clamp"}, True)
    assert c.reach_clamp and c.reach_max == pytest.approx(panto_reach(phys) - REACH_MARGIN)
    assert not SkaterController(model, phys, task, True).reach_clamp


def test_robustness_score_percentile_model_errors_and_push():
    """Phase 1 item 5: the robustness score. `aggregate` p20 / min score a task's repeats at the
    20th percentile / the worst instead of the mean; `mass_error`, `friction_error` and `push`
    perturb each run (drawn after the controller has read the nominal model); absent keys leave
    a run bit-identical to before, so every old score stands."""
    p = json.load(open("configs/archetypes/skater_walker_panto7.json"))["phys"]
    g = encode(p)
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 4.0,
            "pose_noise_deg": 2.0, "servo_model": "real", "servo_lead": "auto", "leg_reach": "clamp"}
    plain = rollout(g, task, 3)
    assert rollout(g, task, 3)["fitness"] == plain["fitness"]
    erred = rollout(g, {**task, "mass_error": 0.1, "friction_error": 0.2, "push": 0.5}, 3)
    assert erred["fitness"] != plain["fitness"]
    assert erred["time_upright"] > 2.0 and erred["fall_dir"] != "drop"   # perturbed, not broken
    assert rollout(g, {**task, "mass_error": 0.1}, 3)["time_upright"] == pytest.approx(4.0, abs=0.02)
    assert rollout(g, {**task, "mass_error": 0.1, "friction_error": 0.2, "push": 0.5}, 3)["fitness"] == erred["fitness"]
    cfg = {"tasks": [{**task, "duration": 2.0, "repeats": 3, "mass_error": 0.1}]}
    e_mean = evaluate(g, cfg, 0)
    e_p20 = evaluate(g, {"tasks": [{**cfg["tasks"][0], "aggregate": "p20"}]}, 0)
    e_min = evaluate(g, {"tasks": [{**cfg["tasks"][0], "aggregate": "min"}]}, 0)
    assert e_mean["w.fitness"] == pytest.approx(e_p20["w.fitness_mean"]) == pytest.approx(e_min["w.fitness_mean"])
    assert e_min["w.fitness"] == e_min["w.fitness_min"] <= e_p20["w.fitness"] <= e_mean["w.fitness"] + 1e-12
    assert "w.upright_min" in e_p20


def test_contact_gait_keys_let_the_strider_walk_and_the_clearance_floor_gates():
    """Phase 1 item 1 on the strider: under the real servo the round-1e walker falls inside 1.3 s with the
    shuffler's contact gait (parked start, measured anchors) and walks 12 s with contact_start 'instant' and
    contact_anchor 'command', the late hold firing on most steps. clear_min gates a low-clearance walk."""
    walker = json.load(open("configs/archetypes/skater_walker_phase1e.json"))["phys"]
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 12.0, "pose_noise_deg": 2.0,
            "servo_model": "real", "servo_lead": "auto", "leg_reach": "clamp", "gait_phase": "contact"}
    shuffler_style = rollout(encode(walker), task, 0)
    assert shuffler_style["time_upright"] < 2.0
    strider_style = rollout(encode(walker), {**task, "contact_start": "instant", "contact_anchor": "command"}, 0)
    assert strider_style["time_upright"] >= 12.0 and strider_style["feet_x_m"] > 1.2
    assert strider_style["contact_events"]["late"] >= 10
    # the clearance floor: the same walk scores lower under a floor above its clearance, the same under one below it
    low = rollout(encode(walker), {**task, "contact_start": "instant", "contact_anchor": "command", "clear_min": 0.08}, 0)
    same = rollout(encode(walker), {**task, "contact_start": "instant", "contact_anchor": "command", "clear_min": 0.01}, 0)
    assert strider_style["swing_clear_m"] < 0.08
    assert low["fitness"] < strider_style["fitness"] - 0.05
    assert abs(same["fitness"] - strider_style["fitness"]) < 1e-6


def test_waist_roll_joint_is_off_by_default_and_a_hung_waist_keeps_the_core_level():
    """The waist (task keys waist_stiffness / waist_damping / waist_pivot): no key, no joint and the old model;
    the core hung from a sprung damped roll pivot at its top (2 N m/rad, 0.1 N m s/rad) rolls under half as much
    as the rigid core on a plain walk of the 48 x 80 walker, which stays upright (measured 2026-11-08: 1.7 vs 5.0 deg)."""
    import math
    import mujoco
    walker = json.load(open("configs/archetypes/skater_walker_shape_long.json"))["phys"]
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 12.0, "pose_noise_deg": 2.0,
            "servo_model": "real", "servo_lead": "auto", "leg_reach": "clamp", "stride_min": 0.1}
    hung = {**task, "waist_stiffness": 2.0, "waist_damping": 0.1, "waist_pivot": 0.14}
    m0, _ = make_model(encode(walker), "flat", np.random.default_rng(0), task)
    m1, _ = make_model(encode(walker), "flat", np.random.default_rng(0), hung)
    assert mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, "waist") == -1
    assert mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_JOINT, "waist") >= 0

    def core_roll(t):
        rolls = []
        def cb(model, data, tt):
            R = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core")].reshape(3, 3)
            if tt > 1.0:
                rolls.append(math.atan2(R[2, 1], R[2, 2]))
        r = rollout(encode(walker), t, 0, frame_cb=cb)
        return r, float(np.sqrt(np.mean(np.square(rolls))))
    r0, c0 = core_roll(task)
    r1, c1 = core_roll(hung)
    assert r1["time_upright"] >= 12.0
    assert c1 < 0.5 * c0


def test_waist_linkage_slaved_to_the_hip_roll_servos_levels_the_core_and_keeps_the_feet_lifting():
    """task key waist_link r: the core's roll on the hip box = r x the mean hip roll angle, a rigid coupling (a
    rocker on both roll horns), no spring. r 0 is the rigid core; r 1 with the pivot at the core's base levels
    the 48 x 80 walker's core (measured 2026-11-08: 3.2 vs 4.9 deg) with no loss of clearance, upright."""
    import math
    import mujoco
    walker = json.load(open("configs/archetypes/skater_walker_shape_long.json"))["phys"]
    task = {"name": "w", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 12.0, "pose_noise_deg": 2.0,
            "servo_model": "real", "servo_lead": "auto", "leg_reach": "clamp", "stride_min": 0.1}

    def run(t):
        rows = []
        def cb(model, data, tt):
            R = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core")].reshape(3, 3)
            q = lambda n: data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
            w = q("waist") if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist") >= 0 else 0.0
            if tt > 1.0:
                rows.append((math.atan2(R[2, 1], R[2, 2]), w - t.get("waist_link", 0.0) * (q("hip_roll_l") + q("hip_roll_r")) / 2))
        r = rollout(encode(walker), t, 0, frame_cb=cb)
        A = np.array(rows)
        return r, float(np.sqrt(np.mean(A[:, 0] ** 2))), float(np.max(np.abs(A[:, 1])))
    r0, c0, _ = run(task)
    r1, c1, slack = run({**task, "waist_link": 1.0})
    assert slack < math.radians(0.2)                       # the coupling holds
    assert r1["time_upright"] >= 12.0
    assert c1 < 0.75 * c0
    assert r1["swing_clear_m"] >= r0["swing_clear_m"] - 0.002
