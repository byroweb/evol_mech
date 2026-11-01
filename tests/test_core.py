import numpy as np
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
    assert model.nu == 7
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
    py = [dict(name=p.name, lo=p.lo, hi=p.hi, default=p.default, group=p.group) for p in PARAMS]
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
