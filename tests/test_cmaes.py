"""CMA-ES optimiser (evolve/cmaes.py): converges on a noisy toy problem with the simulator swapped out."""
import contextlib
import io

import numpy as np

import evol_mech.evolve.cmaes as CM
import evol_mech.evolve.ga as GA
from evol_mech.data.store import RunStore
from evol_mech.skater.spec import PARAMS, N_PARAMS

TARGET = np.full(N_PARAMS, 0.5)
TARGET[:10] = np.linspace(0.2, 0.8, 10)


class _SerialPool:
    def __init__(self, processes):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def map(self, f, jobs, chunksize=1):
        return [f(j) for j in jobs]


def _sphere(args):
    g, cfg, seed = args
    noise = np.random.default_rng(seed).normal(0, cfg["noise"])
    return {"fitness": -100.0 * float(np.sum((g - TARGET) ** 2)) + noise, "sim_steps": 1, "wall_s": 0.0}


def test_cmaes_converges_under_noise_and_keeps_locked_genes(tmp_path, monkeypatch):
    monkeypatch.setattr(CM, "Pool", _SerialPool)
    monkeypatch.setattr(CM, "_eval_one", _sphere)
    monkeypatch.setattr(GA, "_eval_one", _sphere)
    cfg = dict(body="skater", seed=0, population=16, generations=80, workers=1, cma_sigma=0.1, noise=0.3,
               locked=[p.name for p in PARAMS[10:]], final_candidates=3, final_evals=4)
    store = RunStore(tmp_path, "toy", cfg)
    start = CM.get_body("skater").archetype_center(None)
    with contextlib.redirect_stdout(io.StringIO()):
        g, r = CM.run_cmaes(cfg, store)
    store.close()
    assert np.abs(g[:10] - TARGET[:10]).max() < 0.01 < np.abs(start[:10] - TARGET[:10]).max()
    assert np.array_equal(g[10:], start[10:])
    assert "picked_by" in r
