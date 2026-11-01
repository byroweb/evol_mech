import numpy as np
import mujoco
import pytest

from evol_mech.genome import DEFAULT_NORM, decode, encode, random_genome, mutate, crossover, N_PARAMS
from evol_mech.body import linkage, fan_from_params, InvalidBody
from evol_mech.sim import rollout, make_model, TERRAINS


def test_encode_decode_roundtrip():
    phys = decode(DEFAULT_NORM)
    g = encode(phys)
    assert np.allclose(g, DEFAULT_NORM, atol=1e-9)


def test_operators_stay_in_bounds():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = random_genome(rng), random_genome(rng)
        c = mutate(crossover(a, b, rng), rng, 0.3, 0.5)
        assert c.shape == (N_PARAMS,)
        assert c.min() >= 0.0 and c.max() <= 1.0


def test_default_linkage_is_valid_and_rotates():
    p = decode(DEFAULT_NORM)
    assert linkage.is_valid(p)
    path = linkage.foot_path(p)
    assert np.all(np.isfinite(path))
    assert path[:, 1].max() < 0   # foot always below the crank pivot
    # foot is rigid to the coupler: |DP| and |CP| constant over the cycle
    poses = [linkage.solve(p, 2 * np.pi * k / 36) for k in range(36)]
    dp = [np.hypot(q.P[0] - q.D[0], q.P[1] - q.D[1]) for q in poses]
    cp = [np.hypot(q.P[0] - q.C[0], q.P[1] - q.C[1]) for q in poses]
    assert np.ptp(dp) < 1e-9 and np.ptp(cp) < 1e-9


def test_invalid_linkage_is_rejected():
    p = decode(DEFAULT_NORM)
    p["crank_len"] = 0.08
    p["coupler_len"] = 0.05
    p["rocker_len"] = 0.05
    assert not linkage.is_valid(p)
    g = encode(p)
    r = rollout(g, {"name": "t", "terrain": "flat", "objective": "walk", "duration": 1.0})
    assert r["invalid"] and r["fitness"] < 0


def test_fan_model_is_sane():
    p = decode(DEFAULT_NORM)
    f = fan_from_params(p)
    assert 1.0 < f.max_thrust_n < 20.0
    assert 0.02 < f.mass_kg < 0.1


@pytest.mark.parametrize("terrain", TERRAINS)
def test_all_terrains_compile_and_step(terrain):
    model, _ = make_model(DEFAULT_NORM, terrain, np.random.default_rng(0))
    assert model.nu == 6          # 2 cranks, 2 fan pitch, 2 fan thrust (yaw off by default)
    r = rollout(DEFAULT_NORM, {"name": "t", "terrain": terrain, "objective": "walk", "duration": 1.0})
    assert not r["unstable"]
    assert r["sim_steps"] > 0


def test_editor_spec_matches_python_spec():
    """tools/linkage_editor.html embeds a copy of PARAMS; keep them identical."""
    import json, re
    from pathlib import Path
    from evol_mech.genome.spec import PARAMS
    html = Path(__file__).resolve().parents[1].joinpath("tools", "linkage_editor.html").read_text(encoding="utf-8")
    block = re.search(r"const SPEC = (\[.*?\]);", html, re.S).group(1)
    editor = json.loads(block)
    py = [dict(name=p.name, lo=p.lo, hi=p.hi, default=p.default, group=p.group, step=p.step) for p in PARAMS]
    assert editor == py, "run: python -c \"from evol_mech.genome.spec import PARAMS; ...\" and paste into the editor's SPEC"


def test_archetype_roundtrip(tmp_path):
    import json
    from evol_mech.genome import default_archetype, load_archetype, archetype_center, DEFAULT_NORM
    a = default_archetype("t"); a["phys"]["crank_len"] = 0.04
    f = tmp_path / "a.json"; f.write_text(json.dumps(a))
    c = archetype_center(f)
    assert c.shape == DEFAULT_NORM.shape
    assert decode(c)["crank_len"] == pytest.approx(0.04)
    a["phys"]["nope"] = 1.0; f.write_text(json.dumps(a))
    with pytest.raises(ValueError):
        load_archetype(f)


def test_interference_check():
    from evol_mech.body.clearance import interference, is_buildable
    p = decode(DEFAULT_NORM)
    assert interference(p) is None and is_buildable(p)
    low_fans = dict(p, fan_gap=0.0, fan_zf=0.0, fan_diam_in=5.0)
    assert interference(low_fans)[0] == "leg hits fan"
    narrow = dict(p, hip_width=0.06, battery_mah=6000)   # pack wider than the hip allows
    assert interference(narrow)[0] == "leg hits core"
    with pytest.raises(InvalidBody):
        make_model(encode(narrow), "flat", np.random.default_rng(0))


def test_catalogue_steps_and_derived_masses():
    from evol_mech.body.structure import structure_from_params
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = decode(rng.random(N_PARAMS))
        for k in ("fan_diam_in", "fan_blades", "stator_d_mm", "stator_h_mm"):
            assert p[k] == round(p[k])
        assert p["battery_mah"] % 100 == 0
    p = decode(DEFAULT_NORM)
    s = structure_from_params(p)
    assert 0.1 < s.hip_kg < 0.5 and 0.1 < s.core_kg < 0.5 and 0.4 < s.payload_kg < 0.6
    bigger = structure_from_params(dict(p, battery_mah=5000, core_height=0.3))
    assert bigger.payload_kg > s.payload_kg and bigger.core_kg > s.core_kg


def test_waist_yaw_switch():
    g = encode(dict(decode(DEFAULT_NORM), hip_width=0.22, fan_gap=0.10))   # room for core and fans to yaw past the rockers
    m_fixed, _ = make_model(g, "flat", np.random.default_rng(0))
    m_yaw, _ = make_model(g, "flat", np.random.default_rng(0), waist_yaw=True)
    assert m_yaw.nu == m_fixed.nu + 1 and m_yaw.njnt == m_fixed.njnt + 1


def test_foot_types_build_and_battery_fore_aft():
    from evol_mech.body.structure import structure_from_params
    p = decode(DEFAULT_NORM)
    hx, hy, hz = structure_from_params(p).payload_half
    assert hx > hy > hz                      # battery long axis along x (fore/aft), lying flat
    for ft in (0.0, 1.0, 2.0):
        m, _ = make_model(encode(dict(p, foot_type=ft)), "flat", np.random.default_rng(0))
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_l")
        assert m.geom_type[gid] == (mujoco.mjtGeom.mjGEOM_SPHERE if ft == 1.0 else mujoco.mjtGeom.mjGEOM_CAPSULE)


def test_locked_genes_held():
    from evol_mech.genome.ops import random_genome, mutate, crossover
    from evol_mech.evolve.ga import _lock_mask
    from evol_mech.genome.spec import INDEX
    rng = np.random.default_rng(3)
    center = DEFAULT_NORM.copy(); center[INDEX["foot_type"]] = 1.0
    locked = _lock_mask({"locked": ["foot_type", "battery_mah"]})
    for _ in range(10):
        g = random_genome(rng, center=center, locked=locked)
        assert g[locked].tolist() == center[locked].tolist()
    with pytest.raises(ValueError):
        _lock_mask({"locked": ["nope"]})


def test_payload_inside_core_and_floor_grid():
    from evol_mech.body.structure import battery_half, core_half_x, payload_z
    rng = np.random.default_rng(5)
    for _ in range(50):
        p = decode(rng.random(N_PARAMS))
        hx, hy, hz = battery_half(p)
        assert abs(p["payload_x"]) + hx <= core_half_x(p)
        if p["core_height"] / 2 >= hz + 0.005:
            assert abs(payload_z(p)) + hz <= p["core_height"] / 2 + 1e-9
    for kind in ("flat", "cobblestone"):
        m, _ = make_model(DEFAULT_NORM, kind, np.random.default_rng(0))
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "grid") >= 0


def test_only_feet_may_touch_the_floor():
    from evol_mech.sim.rollout import rollout
    m, _ = make_model(DEFAULT_NORM, "flat", np.random.default_rng(0))
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    floor, foot = gid("floor"), gid("foot_l")
    links = [g for g in range(m.ngeom) if m.geom_contype[g] == 0]
    assert links, "expected link/fan geoms with contype 0"
    for g in range(m.ngeom):
        if g == floor:
            continue
        hits_floor = bool(m.geom_contype[g] & m.geom_conaffinity[floor]) or bool(m.geom_contype[floor] & m.geom_conaffinity[g])
        assert hits_floor, f"geom {g} could never touch the floor"
    for g in links:   # links never collide with each other or the feet/body
        assert not (m.geom_contype[g] & m.geom_conaffinity[foot]) and not (m.geom_contype[foot] & m.geom_conaffinity[g])
    # tip the body onto its side: a shin/fan/core touches down and the rollout reports a fall
    import mujoco as mj
    model = m; data = mj.MjData(model); data.qpos[:] = 0; data.qpos[2] = 0.15
    data.qpos[3:7] = [0.7071, 0.7071, 0, 0]   # 90 deg roll
    for _ in range(400):
        mj.mj_step(model, data)
    from evol_mech.sim.rollout import _body_touching_floor
    feet = {gid("foot_l"), gid("foot_r")}
    assert _body_touching_floor(data, {g for g in range(model.ngeom) if g not in feet and g != floor}, floor)


def test_narrow_hip_builds_with_core_following():
    from evol_mech.body.structure import core_half_y, battery_half
    from evol_mech.body.clearance import interference
    p = decode(DEFAULT_NORM)
    narrow = dict(p, hip_width=0.08)
    assert core_half_y(narrow) < core_half_y(p) and core_half_y(narrow) >= battery_half(narrow)[1]
    assert interference(narrow) is None
    m, _ = make_model(encode(narrow), "flat", np.random.default_rng(0))
    assert m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "core")][1] < 0.05
    # a pack too wide for the hip is rejected rather than widening the core into the legs
    assert interference(dict(narrow, hip_width=0.06, battery_mah=6000)) is not None


def test_cross_foot_bar_capped_and_counts_as_foot():
    from evol_mech.body.structure import foot_half_width
    from evol_mech.sim.rollout import rollout
    p = dict(decode(DEFAULT_NORM), foot_type=2, foot_half_width=0.08, hip_width=0.10, foot_radius=0.015)
    assert abs(foot_half_width(p) - (0.05 - 0.015 - 0.005)) < 1e-9
    m, _ = make_model(encode(p), "flat", np.random.default_rng(0))
    bar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_l_bar")
    assert bar >= 0 and m.geom_type[bar] == mujoco.mjtGeom.mjGEOM_CAPSULE
    # standing on cross feet with the cranks still must not register as a fall
    r = rollout(encode(dict(p, crank_speed_hz=0.0)), {"terrain": "flat", "objective": "walk", "duration": 1.0}, seed=0, fans_enabled=False)
    assert not r["fell"]
