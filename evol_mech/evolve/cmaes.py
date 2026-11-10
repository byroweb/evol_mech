"""CMA-ES over the unlocked genes (test plan item 6: the gains tuned on a fixed body).

Same contract as ga.run_ga: the archetype is the start point, cfg 'locked' genes stay at its values, cfg 'gene_bounds'
narrows ranges, every sample of a generation shares that generation's fresh terrain seed (common random numbers, so
the ranking inside a generation compares genomes, not luck), and the winner is picked by the shared final_pick
(mean over several fresh draws). Written out (no `cma` dependency), after Hansen's tutorial (arXiv:1604.00772).

Config keys:
  optimizer      "cmaes" (run.py dispatches on it)
  population     lambda, samples per generation (noise wants more than the default 4 + 3 ln n)
  cma_sigma      initial step size in normalized gene units (default 0.05, the GA's mutation sigma)
  cma_eval_mean  also score the distribution's mean each generation (default true); under noise the mean is the
                 estimate to trust, and the last few means join the final pick
  patience       stop once the mean's score, averaged over the last 5 generations, has not risen by min_delta
                 (default 0.005) for that many generations (default off; counted from generation 4)
Samples outside [0, 1] (or the gene_bounds) are scored clipped, with a penalty of cma_bound_penalty (default 1.0)
times the squared distance clipped away, so the distribution is pushed back inside rather than piling on a wall.
"""
from __future__ import annotations

from multiprocessing import Pool
import time

import numpy as np

from ..bodies import get_body
from ..data.store import RunStore
from .ga import _eval_one, _lock_mask, _bounds, final_pick, KIND_RANDOM, KIND_ELITE


def run_cmaes(cfg: dict, store: RunStore, on_generation=None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(cfg["seed"])
    body = get_body(cfg.get("body", "biped"))
    b_lo, b_hi = _bounds(cfg, body)
    center = np.clip(body.archetype_center(cfg.get("archetype")), b_lo, b_hi)
    free = np.flatnonzero(~_lock_mask(cfg, body))
    n = len(free)
    lam = int(cfg["population"])
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    m = center[free].copy()
    sigma = float(cfg.get("cma_sigma", 0.05))
    C = np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    B, D = np.eye(n), np.ones(n)
    eigen_gen = 0
    eval_mean = bool(cfg.get("cma_eval_mean", True))
    penalty = float(cfg.get("cma_bound_penalty", 1.0))
    patience, min_delta = int(cfg.get("patience", 0)), float(cfg.get("min_delta", 0.005))

    def full(x):
        g = center.copy()
        g[free] = x
        return g

    history: dict[bytes, list[float]] = {}
    genome_of: dict[bytes, np.ndarray] = {}
    mean_keys: list[bytes] = []
    mean_scores: list[float] = []
    best_genome, best_result, best_fit = None, None, -np.inf
    ref, last_gain = -np.inf, 0

    with Pool(processes=cfg["workers"]) as pool:
        for gen in range(cfg["generations"]):
            t0 = time.perf_counter()
            gen_seed = int(rng.integers(0, 2**31 - 1))
            z = rng.standard_normal((lam, n))
            y = z @ (B * D).T                       # y ~ N(0, C)
            xs = m + sigma * y
            gs = [full(x) for x in xs]
            clipped = [np.clip(g, b_lo, b_hi) for g in gs]
            pens = [penalty * float(np.sum((g - c) ** 2)) for g, c in zip(gs, clipped)]
            pop = clipped + ([np.clip(full(m), b_lo, b_hi)] if eval_mean else [])
            results = pool.map(_eval_one, [(g, cfg, gen_seed) for g in pop], chunksize=1)
            wall = time.perf_counter() - t0
            raw = np.array([r["fitness"] for r in results])
            for g, f in zip(pop, raw):
                key = g.tobytes()
                history.setdefault(key, []).append(float(f))
                genome_of[key] = g.copy()
            fit = raw[:lam] - np.array(pens)

            i_best = int(np.argmax(raw))
            if raw[i_best] > best_fit:
                best_fit, best_genome, best_result = float(raw[i_best]), pop[i_best].copy(), results[i_best]
                store.save_best(gen, best_genome, best_result)
            kinds = np.array([KIND_RANDOM] * lam + ([KIND_ELITE] if eval_mean else []))
            row = store.log_generation(gen, np.stack(pop), results, wall, kinds=kinds, gen_seed=gen_seed)
            row["sigma"] = sigma
            if eval_mean:
                mean_keys.append(pop[-1].tobytes())
                mean_scores.append(float(raw[-1]))
                row["mean_genome"] = float(raw[-1])
            if on_generation:
                on_generation(row, pop[i_best], results[i_best])

            # --- update the distribution -----------------------------------
            order = np.argsort(-fit)[:mu]
            m_old = m
            m = m + sigma * (w @ y[order])
            zmean = w @ z[order]
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chi_n < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m - m_old) / sigma
            yw = y[order]
            C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
                 + cmu * (yw.T * w) @ yw)
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            if gen - eigen_gen > lam / (c1 + cmu) / n / 10:
                eigen_gen = gen
                C = np.triu(C) + np.triu(C, 1).T
                ev, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(ev, 1e-20))

            print(f"         sigma {sigma:.4f}  axis ratio {D.max() / D.min():5.1f}"
                  + (f"  mean genome {mean_scores[-1]:+.3f} (5-gen avg {np.mean(mean_scores[-5:]):+.3f})" if eval_mean else ""),
                  flush=True)
            if eval_mean and patience and len(mean_scores) >= 5:    # no reference until five means exist (one lucky
                avg = float(np.mean(mean_scores[-5:]))              # first score stopped item 6's first run at 25)
                if avg > ref + min_delta:
                    ref, last_gain = avg, gen
                elif gen - last_gain >= patience:
                    print(f"stopping at generation {gen}: the mean's 5-generation average has not risen by {min_delta} "
                          f"since generation {last_gain}", flush=True)
                    break

        k = int(cfg.get("final_candidates", 0))
        extra = list(dict.fromkeys(mean_keys[-k:])) if k else []
        best_genome, best_result = final_pick(cfg, store, pool, rng, history, genome_of, extra, gen,
                                              best_genome, best_result, best_fit)
    return best_genome, best_result
