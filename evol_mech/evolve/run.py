"""Entry point:  python -m evol_mech.evolve.run configs/default.json [--generations N]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.store import RunStore
from .ga import run_ga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run-name")
    ap.add_argument("--generations", type=int)
    ap.add_argument("--population", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--archetype", help="archetype JSON from tools/linkage_editor.html")
    ap.add_argument("--lock", help="comma-separated genes held at the archetype value (adds to config 'locked')")
    ap.add_argument("--lock-groups", help="comma-separated gene groups to lock entirely, e.g. geometry,linkage,drivetrain,fan_spec")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    for k in ("run_name", "generations", "population", "workers", "archetype"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    locked = set(cfg.get("locked", []))
    if args.lock:
        locked |= set(args.lock.split(","))
    if args.lock_groups:
        from ..bodies import get_body
        PARAMS = get_body(cfg.get("body", "biped")).PARAMS
        groups = set(args.lock_groups.split(","))
        unknown = groups - {p.group for p in PARAMS}
        if unknown:
            raise SystemExit(f"unknown gene groups {sorted(unknown)}")
        locked |= {p.name for p in PARAMS if p.group in groups}
    if locked:
        cfg["locked"] = sorted(locked)

    store = RunStore(args.runs_dir, cfg["run_name"], cfg)
    print(f"run '{cfg['run_name']}' -> {store.dir}   body={cfg.get('body', 'biped')}  pop={cfg['population']} "
          f"gens={cfg['generations']} workers={cfg['workers']}  "
          f"archetype={cfg.get('archetype') or 'built-in default'}  locked={cfg.get('locked') or '-'}")

    def report(row, best_genome, best_result):
        print(f"gen {row['gen']:4d}  best {row['best']:7.3f}  mean {row['mean']:7.3f}  "
              f"invalid {row['invalid_frac']*100:4.1f}%  "
              f"{row['gen_wall_s']:5.1f}s  {row['steps_per_wall_s']/1e3:7.0f}k steps/s  "
              f"eff {row['parallel_efficiency']:4.1f}x")

    best_genome, best_result = run_ga(cfg, store, report)
    store.close()

    if cfg.get("record_elite_video") and best_genome is not None:
        from ..data.video import render_rollout
        for task in cfg["tasks"]:
            out = store.dir / "videos" / f"best_{task['name']}.mp4"
            render_rollout(best_genome, task, out, fans_enabled=cfg.get("fans_enabled", True),
                           body=cfg.get("body", "biped"))
            print("wrote", out)

    print("best fitness", best_result["fitness"] if best_result else None)
    print("best saved to", store.dir / "best.json")


if __name__ == "__main__":
    main()
