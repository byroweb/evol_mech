"""Evolutionary operators on normalised genomes."""
from __future__ import annotations

import numpy as np

from .spec import DEFAULT_NORM, N_PARAMS, decode
from ..body import linkage


def random_genome(rng: np.random.Generator, spread: float = 0.25, max_tries: int = 100) -> np.ndarray:
    """Default design plus Gaussian noise, clipped, rejection-sampled so the
    linkage is buildable. Starting near a sane default beats uniform random,
    which produces mostly un-standable bodies."""
    for _ in range(max_tries):
        g = np.clip(DEFAULT_NORM + rng.normal(0.0, spread, N_PARAMS), 0.0, 1.0)
        if linkage.is_valid(decode(g)):
            return g
    return DEFAULT_NORM.copy()


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
