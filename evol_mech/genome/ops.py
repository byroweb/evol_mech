"""Evolutionary operators on normalised genomes."""
from __future__ import annotations

import numpy as np

from .spec import DEFAULT_NORM, N_PARAMS, decode
from ..body import linkage


def random_genome(rng: np.random.Generator, spread: float = 0.25, max_tries: int = 100,
                  center: np.ndarray | None = None) -> np.ndarray:
    """A design (the built-in default or an archetype) plus Gaussian noise,
    clipped, rejection-sampled so the linkage is buildable. Starting near a
    sane design beats uniform random, which produces mostly un-standable bodies."""
    c = DEFAULT_NORM if center is None else center
    for _ in range(max_tries):
        g = np.clip(c + rng.normal(0.0, spread, N_PARAMS), 0.0, 1.0)
        if linkage.is_valid(decode(g)):
            return g
    return c.copy()


def mutate(g: np.ndarray, rng: np.random.Generator, sigma: float, rate: float) -> np.ndarray:
    """Per-gene Gaussian mutation with probability `rate`."""
    mask = rng.random(N_PARAMS) < rate
    child = g.copy()
    child[mask] += rng.normal(0.0, sigma, mask.sum())
    return np.clip(child, 0.0, 1.0)


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """BLX-alpha-style blend: child gene drawn uniformly from the parents' span
    extended by 25%. Works well for continuous parameter vectors."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    span = hi - lo
    return np.clip(rng.uniform(lo - 0.25 * span, hi + 0.25 * span), 0.0, 1.0)
