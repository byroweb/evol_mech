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


def _bounds(cfg: dict, body=None) -> tuple[np.ndarray, np.ndarray]:
    """cfg['gene_bounds'] {gene: [min, max]} in physical units narrows a gene's range for one round (e.g. a foot no
    longer than 12 cm) without changing the genome's encoding; returned as normalized lower and upper limits."""
    body = body or get_body(cfg.get("body", "biped"))
    lo, hi = np.zeros(body.N_PARAMS), np.ones(body.N_PARAMS)
    for name, (a, b) in cfg.get("gene_bounds", {}).items():
        if name not in body.INDEX:
            raise ValueError(f"bounded gene {name!r} is not in the {body.name} genome")
        i = body.INDEX[name]
        q = body.PARAMS[i]
        lo[i], hi[i] = (max(0.0, (a - q.lo) / (q.hi - q.lo)), min(1.0, (b - q.lo) / (q.hi - q.lo)))
    return lo, hi


def run_ga(cfg: dict, store: RunStore, on_generation=None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["population"]
    body = get_body(cfg.get("body", "biped"))
    center = body.archetype_center(cfg.get("archetype"))
    locked = _lock_mask(cfg, body)
    b_lo, b_hi = _bounds(cfg, body)
    center = np.clip(center, b_lo, b_hi)
    pop = np.stack([body.random_genome(rng, cfg.get("init_spread", 0.25), center=center,
                                       waist_yaw=cfg.get("waist_yaw", False), locked=locked) for _ in range(n)])
    pop = np.clip(pop, b_lo, b_hi)
    pop[0] = center   # keep the archetype itself in generation 0
    # lineage of the current population: parent indices into the previous
    # generation (-1 = none) and how each individual was made
    parents = np.full((n, 2), -1, dtype=int)
    kinds = np.full(n, KIND_RANDOM, dtype=int)
    best_genome, best_result, best_fit = None, None, -np.inf
    # cfg 'patience' (generations, default off): stop once the best has not risen by more than cfg 'min_delta'
    # (default 0.005) for that long. The best is the luckiest evaluation so far (every generation draws its own
    # starts), so it plateaus; round 1k last gained at generation 11 of 80
    patience, min_delta = int(cfg.get("patience", 0)), float(cfg.get("min_delta", 0.005))
    last_gain_gen, gain_ref = 0, -np.inf
    # every score of every genome (elites are re-scored on each generation's fresh draw), for the final pick
    history: dict[bytes, list[float]] = {}
    genome_of: dict[bytes, np.ndarray] = {}

    with Pool(processes=cfg["workers"]) as pool:
        for gen in range(cfg["generations"]):
            t0 = time.perf_counter()
            gen_seed = int(rng.integers(0, 2**31 - 1))
            results = pool.map(_eval_one, [(g, cfg, gen_seed) for g in pop], chunksize=1)
            wall = time.perf_counter() - t0
            fit = np.array([r["fitness"] for r in results])
            for g_i, f_i in zip(pop, fit):
                key = g_i.tobytes()
                history.setdefault(key, []).append(float(f_i))
                genome_of[key] = g_i.copy()
            last_pop, last_fit = pop.copy(), fit.copy()      # the last generation scored (pop moves on below)

            i_best = int(np.argmax(fit))
            if fit[i_best] > best_fit:
                best_fit, best_genome, best_result = fit[i_best], pop[i_best].copy(), results[i_best]
                store.save_best(gen, best_genome, best_result)

            row = store.log_generation(gen, pop, results, wall, parents=parents, kinds=kinds, gen_seed=gen_seed)
            if on_generation:
                on_generation(row, pop[i_best], results[i_best])
            if fit[i_best] > gain_ref + min_delta:
                gain_ref, last_gain_gen = fit[i_best], gen
            if patience and gen - last_gain_gen >= patience:
                print(f"stopping at generation {gen}: no gain above {min_delta} since generation {last_gain_gen}", flush=True)
                break

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
                child = np.clip(child, b_lo, b_hi)
                next_pop.append(child)
                next_parents.append(par)
                next_kinds.append(kind)
            pop = np.stack(next_pop)
            parents = np.array(next_parents, dtype=int)
            kinds = np.array(next_kinds, dtype=int)

        last = [last_pop[i].tobytes() for i in np.argsort(-last_fit)[:int(cfg.get("final_candidates", 0))]]
        best_genome, best_result = final_pick(cfg, store, pool, rng, history, genome_of, last, gen,
                                              best_genome, best_result, best_fit)

    return best_genome, best_result


def final_pick(cfg: dict, store: RunStore, pool, rng: np.random.Generator, history: dict, genome_of: dict,
               extra: list[bytes], gen: int, best_genome, best_result, best_fit: float):
    """cfg 'final_candidates' K (default off) and 'final_evals' R (default 6): a round's best is its luckiest single
    score, so the top K genomes by mean score so far (plus the keys in `extra`, e.g. the last generation's top K) are
    re-scored on fresh draws until each has R scores, and the best mean is saved as best.json."""
    k_cand, r_evals = int(cfg.get("final_candidates", 0)), int(cfg.get("final_evals", 6))
    if k_cand <= 0 or not history:
        return best_genome, best_result
    means = sorted(history, key=lambda key: -np.mean(history[key]))
    cands = list(dict.fromkeys(means[:k_cand] + list(extra)))
    jobs = [(key, int(rng.integers(0, 2**31 - 1))) for key in cands for _ in range(max(0, r_evals - len(history[key])))]
    more = pool.map(_eval_one, [(genome_of[key], cfg, s) for key, s in jobs], chunksize=1) if jobs else []
    for (key, _), r in zip(jobs, more):
        history[key].append(float(r["fitness"]))
    pick = max(cands, key=lambda key: np.mean(history[key]))
    scores = history[pick]
    best_genome = genome_of[pick].copy()
    best_result = {**(best_result or {}), "fitness": float(np.mean(scores)), "fitness_scores": scores,
                   "fitness_sd": float(np.std(scores)), "picked_by": f"mean of {len(scores)} draws, best of {len(cands)} candidates"}
    store.save_best(gen, best_genome, best_result)
    print(f"final pick: mean {np.mean(scores):+.3f} over {len(scores)} draws (sd {np.std(scores):.3f}) from {len(cands)} "
          f"candidates; luckiest single score was {best_fit:+.3f}", flush=True)
    for key in cands:
        print(f"   candidate mean {np.mean(history[key]):+.3f} sd {np.std(history[key]):.3f} n {len(history[key])}", flush=True)
    return best_genome, best_result
