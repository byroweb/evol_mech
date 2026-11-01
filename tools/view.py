r"""Interactive viewer for a saved genome (or the default body).

    .venv\Scripts\python tools\view.py                      # default body, flat
    .venv\Scripts\python tools\view.py runs\dev\best.json --terrain cobblestone --stand
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from evol_mech.genome import DEFAULT_NORM
from evol_mech.data.store import load_genome
from evol_mech.sim.rollout import make_model, CONTROL_DT, SUBSTEPS
from evol_mech.sim.controller import Controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best", nargs="?", help="path to best.json")
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--stand", action="store_true", help="hold cranks still")
    ap.add_argument("--no-fans", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    genome = load_genome(args.best) if args.best else DEFAULT_NORM
    model, p = make_model(genome, args.terrain, np.random.default_rng(args.seed), park=args.stand)
    data = mujoco.MjData(model)
    ctrl = Controller(model, p, fans_enabled=not args.no_fans, gait_enabled=not args.stand)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 1.5
        v.cam.elevation = -15
        t = 0.0
        while v.is_running():
            data.ctrl[:] = ctrl.act(data, t)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            t += CONTROL_DT
            v.cam.lookat[:] = data.qpos[:3]
            v.sync()
            time.sleep(CONTROL_DT)


if __name__ == "__main__":
    main()
