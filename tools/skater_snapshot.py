r"""Still views and a short clip of a skater body, for design review.

    .venv\Scripts\python tools\skater_snapshot.py                       # default body -> runs/skater_skeleton/
    .venv\Scripts\python tools\skater_snapshot.py runs\skater_dev\best.json --out runs\skater_dev\views

Writes front / side / top / three-quarter stills of the parked crouch, a
labelled summary (masses, tread and fan numbers), an 8 s skate clip and a
frame strip of it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from evol_mech.data.store import load_genome
from evol_mech.skater import (DEFAULT_NORM, decode, make_model, make_data, rollout,
                              structure_from_params, nominal_height)
from evol_mech.body.hardware import fan_from_params, tread_drive_from_params, leg_motor_from_params

W, H = 640, 480


def still(model, data, azimuth, elevation, distance, lookat, path):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = lookat
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, cam)
        img = r.render().copy()
    Image.fromarray(img).save(path)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best", nargs="?")
    ap.add_argument("--out", default="runs/skater_skeleton")
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--terrain", default="flat")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    genome = load_genome(args.best) if args.best else DEFAULT_NORM
    model, p = make_model(genome, args.terrain, np.random.default_rng(0))
    data = make_data(model, p)
    st = structure_from_params(p)
    look = [0.0, 0.0, nominal_height(p) * 0.6]
    d = max(0.9, 2.5 * (nominal_height(p) + p["core_height"]))
    views = {"side": (90, -5), "front": (180, -5), "top": (90, -89), "three_quarter": (135, -20)}
    imgs = {k: still(model, data, az, el, d, look, out / f"{k}.png") for k, (az, el) in views.items()}

    tread, fan, motor = tread_drive_from_params(p), fan_from_params(p), leg_motor_from_params(p)
    summary = {
        "total_kg": round(st.total_kg, 3),
        "hip_kg (shell + 10 QDD motors)": round(st.hip_kg, 3),
        "core_kg": round(st.core_kg, 3), "payload_kg": round(st.payload_kg, 3),
        "per_leg_kg": {"thigh": round(st.thigh_kg, 3), "shin_incl_tread_drive": round(st.shin_kg, 3),
                       "foot": round(st.foot_kg, 3), "fraction_of_total": round(st.leg_fraction, 3)},
        "fan_each_kg": round(st.fan_kg, 3), "fan_max_thrust_n": round(fan.max_thrust_n, 2),
        "leg_motor": {"peak_nm": motor.torque_nm, "kg": round(motor.mass_kg, 3)},
        "tread": {"max_speed_m_s": round(tread.max_speed_m_s, 2), "max_force_n": round(tread.max_force_n, 1),
                  "pulley_torque_nm": round(tread.pulley_torque_nm, 3), "drive_kg": round(tread.mass_kg, 3)},
        "hip_height_m": round(nominal_height(p), 3),
        "phys": p,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # composite of the four stills
    sheet = Image.new("RGB", (2 * W, 2 * H), "white")
    for k, (i, j) in zip(views, [(0, 0), (1, 0), (0, 1), (1, 1)]):
        sheet.paste(Image.fromarray(imgs[k]), (i * W, j * H))
        ImageDraw.Draw(sheet).text((i * W + 10, j * H + 10), k, fill=(0, 0, 0))
    sheet.save(out / "views.png")

    # skate clip + strip
    frames, state = [], {"r": None, "next": 0.0}
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.distance, cam.elevation, cam.azimuth = d, -12, 120
    def cb(m, dd, t):
        if state["r"] is None:
            state["r"] = mujoco.Renderer(m, H, W)
            cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hip")
        if t >= state["next"]:
            state["r"].update_scene(dd, cam)
            frames.append(state["r"].render().copy())
            state["next"] += 1 / 30
    task = {"name": "skate", "terrain": args.terrain, "objective": "skate", "speed": args.speed, "duration": 8.0}
    res = rollout(genome, task, 0, frame_cb=cb, stop_on_fall=False)
    if state["r"] is not None:
        state["r"].close()
    imageio.mimwrite(str(out / "skate.mp4"), frames, fps=30, codec="libx264", quality=7)
    pick = frames[::30][:9]
    strip = Image.new("RGB", (len(pick) * W // 2, H // 2), "white")
    for i, f in enumerate(pick):
        strip.paste(Image.fromarray(f).resize((W // 2, H // 2)), (i * W // 2, 0))
        ImageDraw.Draw(strip).text((i * W // 2 + 6, 6), f"{i}s", fill=(0, 0, 0))
    strip.save(out / "skate_strip.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "phys"}, indent=2))
    print("skate:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                     if k in ("fitness", "time_upright", "distance_x", "drift_y", "mean_speed", "fell", "rms_accel")})
    print("wrote", out)


if __name__ == "__main__":
    main()
