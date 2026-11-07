"""Motor angles against time for one skater episode, one panel per motor.

    python tools/joint_trace.py configs/archetypes/skater_walker_evolved.json --objective walk --duration 4 --out trace.png
    python tools/joint_trace.py runs/skater_walk_gait2/best.json ...

The input is an archetype JSON ("phys") or a run's best.json ("phys" or "genome").
Panels (left leg solid, right leg dashed): hip roll servo, thigh servo, shin
servo (its horn angle: the knee hinge follows the four-bar), measured knee
hinge, passive ankle pitch and roll, fan pitch, fan throttle. Commanded angle in
colour, measured joint angle in grey. Also prints gait-smoothness metrics:
realized stride, servo angular-acceleration RMS and the share of each servo
command's power above 2.5 x the step rate (see docs/skater_review.md).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco                                           # noqa: E402
from evol_mech.skater.spec import encode                # noqa: E402
from evol_mech.skater.rollout import make_model, make_data, SUBSTEPS   # noqa: E402
from evol_mech.skater.controller import SkaterController, CONTROL_DT, quat_to_roll_pitch   # noqa: E402

SERVOS = ("hip_roll", "hip_pitch", "horn")            # the three leg servos
JOINTS = ("hip_roll", "hip_pitch", "knee", "ankle", "ankle_roll", "fan_pitch")
PANELS = [("hip_roll", "hip roll servo (deg)"), ("hip_pitch", "thigh servo (deg)"), ("horn", "shin servo horn (deg)"),
          ("knee", "knee hinge, measured (deg)"), ("ankle", "ankle pitch, passive (deg)"),
          ("ankle_roll", "ankle roll, passive (deg)"), ("fan_pitch", "fan pitch (deg)"), ("thr", "fan throttle")]


def load_genome(path: str) -> np.ndarray:
    d = json.load(open(path))
    if "phys" in d:
        return encode(d["phys"])
    return np.asarray(d["genome"], dtype=float)


def trace(genome: np.ndarray, task: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    model, p = make_model(genome, task["terrain"], rng, task)
    data = make_data(model, p)
    ctrl = SkaterController(model, p, task, True)
    jq = {(j, s): model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{s}")]
          for s in ("l", "r") for j in JOINTS}
    log: dict = {k: [] for k in ("t", "x", "y", "z", "roll", "pitch")}
    for s in ("l", "r"):
        for j in JOINTS:
            log[(j, s)] = []
        for j in ("hip_roll", "hip_pitch", "horn", "fan_pitch", "thr"):
            log[("cmd_" + j, s)] = []
    n = int(float(task["duration"]) / CONTROL_DT)
    for k in range(n):
        t = k * CONTROL_DT
        data.ctrl[:] = ctrl.act(data, t)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        r, pt = quat_to_roll_pitch(data.qpos[3:7])
        log["t"].append(t)
        log["x"].append(float(data.qpos[0]))
        log["y"].append(float(data.qpos[1]))
        log["z"].append(float(data.qpos[2]))
        log["roll"].append(r)
        log["pitch"].append(pt)
        for s in ("l", "r"):
            for j in JOINTS:
                log[(j, s)].append(float(data.qpos[jq[(j, s)]]))
            log[("cmd_hip_roll", s)].append(float(data.ctrl[ctrl.idx[f"hip_roll_{s}"]]))
            log[("cmd_hip_pitch", s)].append(float(data.ctrl[ctrl.idx[f"hip_pitch_{s}"]]))
            log[("cmd_horn", s)].append(ctrl.horn_cmd[s])
            log[("cmd_fan_pitch", s)].append(float(data.ctrl[ctrl.idx[f"fan_pitch_{s}"]]))
            log[("cmd_thr", s)].append(float(data.ctrl[ctrl.idx[f"fan_thrust_{s}"]]))
        if abs(r) > 1.0 or abs(pt) > 1.0:
            break
    out = {k: np.asarray(v) for k, v in log.items()}
    out["p"] = p
    return out


def smoothness(log: dict, p: dict) -> dict:
    """Candidate grace metrics on the servo commands."""
    t = log["t"]
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out = {"duration_s": dur, "progress_m": float(log["x"][-1] - log["x"][0]), "step_hz": p["step_hz"]}
    out["strides"] = p["step_hz"] * dur
    out["stride_m"] = out["progress_m"] / out["strides"] if out["strides"] else 0.0   # realized distance per step
    acc, hf = [], []
    for s in ("l", "r"):
        for j in SERVOS:
            q = log[("cmd_" + j, s)]
            if len(q) < 8:
                continue
            a = np.diff(q, 2) / CONTROL_DT ** 2                  # commanded angular acceleration
            acc.append(math.sqrt(float(np.mean(a ** 2))))
            spec = np.abs(np.fft.rfft(q - q.mean())) ** 2
            f = np.fft.rfftfreq(len(q), CONTROL_DT)
            tot = float(spec.sum()) or 1.0
            hf.append(float(spec[f > 2.5 * p["step_hz"]].sum()) / tot)
    out["servo_accel_rms_rad_s2"] = float(np.mean(acc)) if acc else 0.0
    out["servo_power_above_2.5x_step"] = float(np.mean(hf)) if hf else 0.0
    out["hip_z_sway_cm"] = float(np.std(log["z"])) * 100
    out["body_roll_rms_deg"] = math.degrees(float(np.sqrt(np.mean(log["roll"] ** 2))))
    out["body_pitch_rms_deg"] = math.degrees(float(np.sqrt(np.mean(log["pitch"] ** 2))))
    return out


def plot(log: dict, title: str, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = log["t"]
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 2.0 * len(PANELS)), sharex=True)
    for ax, (key, label) in zip(axes, PANELS):
        for s, ls in (("l", "-"), ("r", "--")):
            if ("cmd_" + key, s) in log:
                y = log[("cmd_" + key, s)]
                ax.plot(t, y if key == "thr" else np.degrees(y), ls, lw=1.2, label=f"{s} command")
            if (key, s) in log:
                ax.plot(t, np.degrees(log[(key, s)]), ls, lw=0.9, color="0.55", label=f"{s} measured")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=4)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="archetype JSON or a run's best.json")
    ap.add_argument("--objective", default="walk")
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-hz-max", type=float, help="cap the walk cadence as a task may (step_hz_max)")
    ap.add_argument("--out", default="joint_trace.png")
    a = ap.parse_args()
    task = {"name": a.objective, "objective": a.objective, "terrain": a.terrain, "duration": a.duration, "speed": a.speed}
    if a.step_hz_max:
        task["step_hz_max"] = a.step_hz_max
    log = trace(load_genome(a.source), task, a.seed)
    p = log.pop("p")
    for k, v in smoothness(log, p).items():
        print(f"  {k:30s} {v:8.3f}")
    plot(log, f"{Path(a.source).name}  {a.objective} on {a.terrain}", a.out)


if __name__ == "__main__":
    main()
