"""Generational GA with elitism, tournament selection, blend crossover and
Gaussian mutation, evaluated on a multiprocessing pool.

Every individual in a generation sees the same terrain seed so fitness is
comparable within the generation; the seed changes each generation so bodies
cannot overfit one terrain instance.
"""
from __future__ import annotations

from multiprocessing import Pool
import time

import numpy as np

from ..genome import random_genome, mutate, crossover, N_PARAMS
from ..genome.archetype import archetype_center
from ..sim.rollout import evaluate
from ..data.store import RunStore


def _eval_one(args):
    genome, cfg, seed = args
    return evaluate(genome, cfg, seed)


def _tournament(fit: np.ndarray, rng: np.random.Generator, k: int) -> int:
    idx = rng.integers(0, len(fit), k)
    return int(idx[np.argmax(fit[idx])])


def run_ga(cfg: dict, store: RunStore, on_generation=None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["population"]
    center = archetype_center(cfg.get("archetype"))
    pop = np.stack([random_genome(rng, cfg.get("init_spread", 0.25), center=center,
                                  waist_yaw=cfg.get("waist_yaw", False)) for _ in range(n)])
    pop[0] = center   # keep the archetype itself in generation 0
    best_genome, best_result, best_fit = None, None, -np.inf

    with Pool(processes=cfg["workers"]) as pool:
        for gen in range(cfg["generations"]):
            t0 = time.perf_counter()
            gen_seed = int(rng.integers(0, 2**31 - 1))
            results = pool.map(_eval_one, [(g, cfg, gen_seed) for g in pop], chunksize=1)
            wall = time.perf_counter() - t0
            fit = np.array([r["fitness"] for r in results])

            i_best = int(np.argmax(fit))
            if fit[i_best] > best_fit:
                best_fit, best_genome, best_result = fit[i_best], pop[i_best].copy(), results[i_best]
                store.save_best(gen, best_genome, best_result)

            row = store.log_generation(gen, pop, results, wall)
            if on_generation:
                on_generation(row, pop[i_best], results[i_best])

            # --- next generation --------------------------------------------
            order = np.argsort(-fit)
            next_pop = [pop[i].copy() for i in order[: cfg["elites"]]]
            while len(next_pop) < n:
                a = pop[_tournament(fit, rng, cfg["tournament_k"])]
                b = pop[_tournament(fit, rng, cfg["tournament_k"])]
                child = crossover(a, b, rng) if rng.random() < 0.7 else a.copy()
                next_pop.append(mutate(child, rng, cfg["mutation_sigma"], cfg["mutation_rate"]))
            pop = np.stack(next_pop)

    return best_genome, best_result
