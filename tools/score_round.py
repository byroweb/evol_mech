r"""Score a finished GA round the way the skater rounds are judged (docs/skater_test_plan.md):

    .venv\Scripts\python tools\score_round.py phase1h phase1e

saves the round's best as configs/archetypes/skater_walker_<round>.json, lists the genes that moved from the seed
archetype, then scores seed and winner on the round's own tasks with a base seed the GA never saw (7000 + task
index + 100 x repeat) and on the plain four tilted starts under the real servo (the round's gait keys carried
over), and prints the 20th-percentile fitness per task, the weighted total, and per-start upright time with the
fall direction (f/b/l/r/n). The GA's own best fitness has overstated every round; this number is the one to
report. Multiprocessing: run it as a file, never through stdin (Windows spawn)."""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from evol_mech.skater.spec import encode, decode  # noqa: E402

ROUND, SEED = sys.argv[1], sys.argv[2]
A = "configs/archetypes/"
cfg = json.load(open(f"configs/skater_walk_{ROUND}.json"))
TASKS = {t["name"]: t for t in cfg["tasks"]}
WALK0 = next(t for t in cfg["tasks"] if t["objective"] == "walk")
GAIT_KEYS = ("gait_phase", "foot_placement", "contact_start", "contact_anchor", "clear_min", "stride_min")
PLAIN = {"name": "plain", "terrain": "flat", "objective": "walk", "speed": 0.4, "step_hz_max": 2.5, "duration": 12.0, "pose_noise_deg": 2.0,
         "servo_model": "real", "servo_lead": "auto", "leg_reach": "clamp", **{k: WALK0[k] for k in GAIT_KEYS if k in WALK0}}
KEYS = ("fitness", "grace", "feet_x_m", "stride_m", "swing_clear_m", "roll_rms", "fan_duty", "servo_sat_frac", "time_upright", "fall_dir")


def save_best():
    best = json.load(open(f"runs/skater_walk_{ROUND}/best.json"))
    g = np.array(best["genome"] if "genome" in best else best["best_genome"])
    phys = decode(g)
    base = decode(encode(json.load(open(A + f"skater_walker_{SEED}.json"))["phys"]))
    note = f"configs/skater_walk_{ROUND}.json best, from skater_walker_{SEED}: " + cfg.get("_note", "").split(". Seed check")[0]
    json.dump({"name": f"skater_walker_{ROUND}", "note": note, "phys": phys}, open(A + f"skater_walker_{ROUND}.json", "w"), indent=1)
    print(f"GA fitness {best.get('fitness', best.get('result', {}).get('fitness'))} (generation {best.get('gen')})")
    moved = sorted([(k, base[k], phys[k]) for k in phys if abs(phys[k] - base[k]) > 1e-9], key=lambda m: -abs(m[2] - m[1]) / (abs(m[1]) + 1e-9))
    print(f"{len(moved)} genes moved; the 15 largest relative moves:")
    for k, a, b in moved[:15]:
        print(f"  {k:14s} {a:9.4f} -> {b:9.4f}")


def one(a):
    name, tname, rep = a
    from evol_mech.skater.rollout import rollout
    t = TASKS.get(tname, PLAIN)
    g = encode(json.load(open(A + f"skater_walker_{name}.json"))["phys"])
    seed = (7000 + list(TASKS).index(tname) + 100 * rep) if tname in TASKS else rep
    r = rollout(g, t, seed, fans_enabled=cfg.get("fans_enabled", True), fan_energy_cost=cfg.get("fan_energy_cost", 0.2),
                motor_energy_cost=cfg.get("motor_energy_cost", 0.002), leg_mass_penalty=cfg.get("leg_mass_penalty"))   # the GA's own costs
    return name, tname, rep, {k: r.get(k, float("nan")) for k in KEYS}


if __name__ == "__main__":
    save_best()
    names = (SEED, ROUND)
    jobs = [(n, tn, rep) for n in names for tn, t in TASKS.items() for rep in range(int(t["repeats"]))] + [(n, "plain", s) for n in names for s in range(4)]
    with Pool(int(cfg.get("workers", 12))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    for n in names:
        print(f"\n{n}  (gait keys " + ", ".join(f"{k}={PLAIN[k]}" for k in GAIT_KEYS if k in PLAIN) + ")")
        total = 0.0
        for tn in list(TASKS) + ["plain"]:
            rows = sorted([r for r in res if r[0] == n and r[1] == tn], key=lambda r: r[2])
            f = np.array([d["fitness"] for _, _, _, d in rows])
            p20 = float(np.percentile(f, 20))
            if tn in TASKS:
                total += TASKS[tn]["weight"] * p20
            mean = {k: float(np.nanmean([d[k] for *_, d in rows])) for k in KEYS if k != "fall_dir"}
            print(f"  {tn:16s} p20 {p20:+.3f} " + " ".join(f"{k[:9]}={v:.3f}" for k, v in mean.items() if k != "fitness")
                  + "  upright " + " ".join(f"{d['time_upright']:.1f}{d['fall_dir'][0]}" for *_, d in rows))
        print(f"  weighted p20 score on unseen seeds {total:+.3f}")
