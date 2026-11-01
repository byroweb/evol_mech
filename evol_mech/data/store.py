"""Run directory layout and result logging.

runs/<run_name>/
    config.json          frozen copy of the config used
    generations.jsonl    one line per generation: summary stats + throughput
    population/gen_XXXX.npz   genomes + fitness for every individual
    best.json            best-so-far genome (normalised + decoded) and metrics
    videos/              elite replays, if enabled
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from ..genome.spec import decode


class RunStore:
    def __init__(self, root: Path | str, run_name: str, cfg: dict):
        self.dir = Path(root) / run_name
        (self.dir / "population").mkdir(parents=True, exist_ok=True)
        (self.dir / "videos").mkdir(exist_ok=True)
        with open(self.dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        self.gen_log = open(self.dir / "generations.jsonl", "a")
        self.t0 = time.time()

    def log_generation(self, gen: int, genomes: np.ndarray, results: list[dict],
                       wall_s: float) -> dict:
        fit = np.array([r["fitness"] for r in results])
        steps = sum(r["sim_steps"] for r in results)
        cpu_s = sum(r["wall_s"] for r in results)
        row = {
            "gen": gen,
            "t": time.time() - self.t0,
            "best": float(fit.max()),
            "mean": float(fit.mean()),
            "median": float(np.median(fit)),
            "invalid_frac": float(np.mean([r.get("invalid", False) for r in results])),
            "sim_steps": int(steps),
            "gen_wall_s": wall_s,
            "worker_cpu_s": cpu_s,
            "steps_per_wall_s": steps / wall_s if wall_s > 0 else 0.0,
            "steps_per_cpu_s": steps / cpu_s if cpu_s > 0 else 0.0,
            "parallel_efficiency": (cpu_s / wall_s) if wall_s > 0 else 0.0,
        }
        self.gen_log.write(json.dumps(row) + "\n")
        self.gen_log.flush()
        np.savez_compressed(self.dir / "population" / f"gen_{gen:04d}.npz",
                            genomes=genomes, fitness=fit)
        return row

    def save_best(self, gen: int, genome: np.ndarray, result: dict) -> None:
        with open(self.dir / "best.json", "w") as f:
            json.dump({"gen": gen, "genome": genome.tolist(), "phys": decode(genome),
                       "result": result}, f, indent=2)

    def close(self):
        self.gen_log.close()


def load_genome(path: Path | str) -> np.ndarray:
    with open(path) as f:
        return np.array(json.load(f)["genome"])
