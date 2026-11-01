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
    low_fans = dict(p, fan_gap=0.0, fan_z=0.0, fan_diam_in=5.0)
    assert interference(low_fans)[0] == "leg hits fan"
    narrow = dict(p, hip_width=0.08)
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
    g = encode(dict(decode(DEFAULT_NORM), hip_width=0.22))   # room for the core to yaw past the rockers
    m_fixed, _ = make_model(g, "flat", np.random.default_rng(0))
    m_yaw, _ = make_model(g, "flat", np.random.default_rng(0), waist_yaw=True)
    assert m_yaw.nu == m_fixed.nu + 1 and m_yaw.njnt == m_fixed.njnt + 1


def test_foot_types_build_and_battery_fore_aft():
    from evol_mech.body.structure import structure_from_params
    p = decode(DEFAULT_NORM)
    hx, hy, hz = structure_from_params(p).payload_half
    assert hx > hy > hz                      # battery long axis along x (fore/aft), lying flat
    for ft in (0.0, 1.0):
        m, _ = make_model(encode(dict(p, foot_type=ft)), "flat", np.random.default_rng(0))
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_l")
        assert m.geom_type[gid] == (mujoco.mjtGeom.mjGEOM_SPHERE if ft else mujoco.mjtGeom.mjGEOM_CAPSULE)


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
