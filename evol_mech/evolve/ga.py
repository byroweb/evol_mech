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

from ..bodies import get_body
from ..data.store import RunStore


# how an individual came to be (stored per generation for lineage views)
KIND_RANDOM, KIND_ELITE, KIND_MUTANT, KIND_CROSSOVER = 0, 1, 2, 3
KIND_NAMES = {KIND_RANDOM: "random", KIND_ELITE: "elite", KIND_MUTANT: "mutant", KIND_CROSSOVER: "crossover"}


def _eval_one(args):
    genome, cfg, seed = args
    return get_body(cfg.get("body", "biped")).evaluate(genome, cfg, seed)


def _tournament(fit: np.ndarray, rng: np.random.Generator, k: int) -> int:
    idx = rng.integers(0, len(fit), k)
    return int(idx[np.argmax(fit[idx])])


def _lock_mask(cfg: dict, body=None) -> np.ndarray:
    """Genes listed in cfg['locked'] are held at the archetype's value."""
    body = body or get_body(cfg.get("body", "biped"))
    mask = np.zeros(body.N_PARAMS, dtype=bool)
    for name in cfg.get("locked", []):
        if name not in body.INDEX:
            raise ValueError(f"locked gene {name!r} is not in the {body.name} genome")
        mask[body.INDEX[name]] = True
    return mask


def run_ga(cfg: dict, store: RunStore, on_generation=None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["population"]
    body = get_body(cfg.get("body", "biped"))
    center = body.archetype_center(cfg.get("archetype"))
    locked = _lock_mask(cfg, body)
    pop = np.stack([body.random_genome(rng, cfg.get("init_spread", 0.25), center=center,
                                       waist_yaw=cfg.get("waist_yaw", False), locked=locked) for _ in range(n)])
    pop[0] = center   # keep the archetype itself in generation 0
    # lineage of the current population: parent indices into the previous
    # generation (-1 = none) and how each individual was made
    parents = np.full((n, 2), -1, dtype=int)
    kinds = np.full(n, KIND_RANDOM, dtype=int)
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

            row = store.log_generation(gen, pop, results, wall, parents=parents, kinds=kinds, gen_seed=gen_seed)
            if on_generation:
                on_generation(row, pop[i_best], results[i_best])

            # --- next generation --------------------------------------------
            order = np.argsort(-fit)
            next_pop = [pop[i].copy() for i in order[: cfg["elites"]]]
            next_parents = [(int(i), -1) for i in order[: cfg["elites"]]]
            next_kinds = [KIND_ELITE] * len(next_pop)
            while len(next_pop) < n:
                ia = _tournament(fit, rng, cfg["tournament_k"])
                ib = _tournament(fit, rng, cfg["tournament_k"])
                a, b = pop[ia], pop[ib]
                if rng.random() < 0.7:
                    child, par, kind = body.crossover(a, b, rng), (ia, ib), KIND_CROSSOVER
                else:
                    child, par, kind = a.copy(), (ia, -1), KIND_MUTANT
                child = body.mutate(child, rng, cfg["mutation_sigma"], cfg["mutation_rate"])
                child[locked] = center[locked]
                next_pop.append(child)
                next_parents.append(par)
                next_kinds.append(kind)
            pop = np.stack(next_pop)
            parents = np.array(next_parents, dtype=int)
            kinds = np.array(next_kinds, dtype=int)

    return best_genome, best_result
