r"""Draw one leg's 4-bar linkage at several crank angles plus the foot path,
labelled as in the design sketch (A, B, C, D, P). Writes a PNG.

    .venv\Scripts\python tools\plot_linkage.py                       # default body
    .venv\Scripts\python tools\plot_linkage.py runs\dev\best.json -o leg.png
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evol_mech.genome import DEFAULT_NORM, decode
from evol_mech.body import linkage
from evol_mech.data.store import load_genome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best", nargs="?", help="path to best.json (default: default body)")
    ap.add_argument("-o", "--out", default="linkage.png")
    ap.add_argument("--poses", type=int, default=8)
    args = ap.parse_args()

    g = load_genome(args.best) if args.best else DEFAULT_NORM
    p = decode(g)
    valid = linkage.is_valid(p)
    path = linkage.foot_path(p, 180)
    B = (p["pivot_b_dx"], p["pivot_b_dz"])

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.plot(path[:, 0], path[:, 1], "b-", lw=2, label="foot path (P)")
    for k in range(args.poses):
        th = 2 * math.pi * k / args.poses
        q = linkage.solve(p, th)
        if q is None:
            continue
        alpha = 1.0 if k == 0 else 0.25
        ax.plot([0, q.D[0]], [0, q.D[1]], "r-", lw=2, alpha=alpha)                 # crank AD
        ax.plot([B[0], q.C[0]], [B[1], q.C[1]], "g-", lw=2, alpha=alpha)           # rocker BC
        ax.plot([q.D[0], q.C[0], q.P[0], q.D[0]],
                [q.D[1], q.C[1], q.P[1], q.D[1]], "k-", lw=1.5, alpha=alpha)       # coupler triangle
    ax.plot([0, B[0]], [0, B[1]], "k--", lw=1, label="frame AB")
    q0 = linkage.solve(p, 0.0)
    for name, pt in (("A", (0, 0)), ("B", B), ("C", q0.C), ("D", q0.D), ("P", q0.P)):
        ax.plot(*pt, "ko", ms=4)
        ax.annotate(name, pt, textcoords="offset points", xytext=(5, 5))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("z up [m]")
    ax.set_title(f"valid crank-rocker: {valid}   crank {p['crank_len']*1000:.0f}mm  "
                 f"coupler {p['coupler_len']*1000:.0f}mm  rocker {p['rocker_len']*1000:.0f}mm")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out, "| valid:", valid,
          "| foot path x %.3f..%.3f z %.3f..%.3f" % (np.nanmin(path[:, 0]), np.nanmax(path[:, 0]),
                                                      np.nanmin(path[:, 1]), np.nanmax(path[:, 1])))


if __name__ == "__main__":
    main()
