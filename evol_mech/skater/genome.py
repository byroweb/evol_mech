"""Evolutionary operators and archetypes for the skater genome.

Same operators as `genome/ops.py` but sized by the vector they are given, and
rejection-sampled against the skater's buildability check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .spec import DEFAULT_NORM, N_PARAMS, INDEX, LO, HI, decode, encode
from .structure import is_buildable


def random_genome(rng: np.random.Generator, spread: float = 0.25, max_tries: int = 100,
                  center: np.ndarray | None = None, locked: np.ndarray | None = None,
                  **_ignored) -> np.ndarray:
    c = DEFAULT_NORM if center is None else center
    for _ in range(max_tries):
        g = np.clip(c + rng.normal(0.0, spread, N_PARAMS), 0.0, 1.0)
        if locked is not None:
            g[locked] = c[locked]
        if is_buildable(decode(g)):
            return g
    return c.copy()


def mutate(g: np.ndarray, rng: np.random.Generator, sigma: float, rate: float) -> np.ndarray:
    mask = rng.random(g.shape[0]) < rate
    child = g.copy()
    child[mask] += rng.normal(0.0, sigma, int(mask.sum()))
    return np.clip(child, 0.0, 1.0)


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    span = hi - lo
    return np.clip(rng.uniform(lo - 0.25 * span, hi + 0.25 * span), 0.0, 1.0)


def load_archetype(path: str | Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    phys = d.get("phys", {})
    unknown = sorted(set(phys) - set(INDEX))
    if unknown:
        raise ValueError(f"archetype {path}: unknown parameters {unknown}")
    for k, v in phys.items():
        i = INDEX[k]
        if not (LO[i] - 1e-9 <= v <= HI[i] + 1e-9):
            raise ValueError(f"archetype {path}: {k}={v} outside genome bounds [{LO[i]}, {HI[i]}]")
    d["phys"] = phys
    return d


def archetype_center(path: str | Path | None) -> np.ndarray:
    if not path:
        return DEFAULT_NORM.copy()
    return encode(load_archetype(path)["phys"])


def default_archetype(name: str = "skater_default") -> dict:
    return {"name": name, "body": "skater", "tags": {}, "phys": decode(DEFAULT_NORM)}
