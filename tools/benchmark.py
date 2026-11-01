r"""Throughput benchmark: single-process steps/sec, then pool scaling.

    .venv\Scripts\python tools\benchmark.py [--workers 1 4 8 12 16 20]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
import time

import numpy as np

from evol_mech.genome import DEFAULT_NORM, random_genome
from evol_mech.sim.rollout import rollout

TASK = {"name": "bench", "terrain": "flat", "objective": "walk", "duration": 4.0}
CFG = {"tasks": [TASK], "fans_enabled": True, "fan_energy_cost": 0.02}


def _eval(args):
    g, seed = args
    return rollout(g, TASK, seed=seed, stop_on_fall=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20])
    ap.add_argument("--n", type=int, default=64, help="rollouts per pool test")
    args = ap.parse_args()

    # single process: a body that stands long enough to measure full-length episodes
    r = rollout(DEFAULT_NORM, TASK, seed=0, stop_on_fall=False)
    print(f"single process: {r['steps_per_s']/1e3:7.1f}k steps/s  ({r['sim_steps']} steps, "
          f"{r['wall_s']*1e3:.1f} ms wall, upright {r['time_upright']:.2f}s)")

    rng = np.random.default_rng(0)
    genomes = [random_genome(rng) for _ in range(args.n)]
    print(f"\npool scaling, {args.n} random rollouts of {TASK['duration']}s each:")
    print(f"{'workers':>8} {'wall s':>8} {'rollouts/s':>11} {'steps/s':>10} {'speedup':>8}")
    base = None
    for w in args.workers:
        with Pool(w) as pool:
            pool.map(_eval, [(genomes[0], 0)] * w)  # warm up workers
            t0 = time.perf_counter()
            res = pool.map(_eval, [(g, i) for i, g in enumerate(genomes)], chunksize=1)
            wall = time.perf_counter() - t0
        steps = sum(x["sim_steps"] for x in res)
        base = base or wall
        print(f"{w:8d} {wall:8.2f} {args.n/wall:11.1f} {steps/wall/1e3:9.0f}k {base/wall:8.2f}x")


if __name__ == "__main__":
    main()
