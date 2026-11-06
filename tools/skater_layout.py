r"""Annotated actuator-layout drawing of a skater body (side and front views).

    .venv\Scripts\python tools\skater_layout.py                      # -> runs/skater_skeleton/servo_layout.png
    .venv\Scripts\python tools\skater_layout.py runs\x\best.json --out layout.png

Pixel positions come from a segmentation render (geom id per pixel) fitted to
the geoms' world positions, so labels land on the parts without camera maths.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evol_mech.data.store import load_genome
from evol_mech.skater import DEFAULT_NORM, make_model, make_data, nominal_height

W, H = 640, 480
PX, PY = 260, 90           # margins around each render for labels
PW, PH = W + 2 * PX, H + 2 * PY
HIP = (200, 60, 60)        # motors in the hip cluster
SHIN = (200, 120, 0)       # tread drive on the shin
PASSIVE = (40, 90, 200)    # unpowered / structure
FAN = (120, 60, 160)


def render(model, data, azimuth, elevation, distance, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = lookat
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, cam)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(data, cam)
        seg = r.render().copy()
    return rgb, seg[:, :, 0].astype(int)   # geom id per pixel (-1 background)


def fit_projection(model, data, seg):
    """Least-squares affine world -> pixel from the visible geoms' centroids."""
    pts, pix = [], []
    for g in range(model.ngeom):
        ys, xs = np.nonzero(seg == g)
        if len(xs) < 30:
            continue
        pts.append(data.geom_xpos[g])
        pix.append((xs.mean(), ys.mean()))
    A = np.hstack([np.array(pts), np.ones((len(pts), 1))])
    M, *_ = np.linalg.lstsq(A, np.array(pix), rcond=None)
    return lambda p: tuple((np.append(p, 1.0) @ M).tolist())


def joint_pos(model, data, name):
    return data.xanchor[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)].copy()


def geom_pos(model, data, name):
    return data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)].copy()


def annotate(img, proj, items, font):
    """items: (world_point, label_lines, colour, (dx, dy) label offset in px)"""
    d = ImageDraw.Draw(img)
    for p, lines, col, (dx, dy) in items:
        x, y = proj(p)
        lx, ly = x + dx, y + dy
        d.line([(x, y), (lx, ly)], fill=col, width=2)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=col, width=3)
        tw = max(d.textlength(t, font=font) for t in lines)
        th = 16 * len(lines)
        bx = lx if dx >= 0 else lx - tw
        d.rectangle([bx - 4, ly - th / 2 - 3, bx + tw + 4, ly + th / 2 + 3], fill=(255, 255, 255), outline=col)
        for i, t in enumerate(lines):
            d.text((bx, ly - th / 2 + 16 * i), t, fill=col, font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best", nargs="?")
    ap.add_argument("--out", default="runs/skater_skeleton/servo_layout.png")
    args = ap.parse_args()
    genome = load_genome(args.best) if args.best else DEFAULT_NORM
    model, p = make_model(genome, "flat", np.random.default_rng(0))
    data = make_data(model, p)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        big = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = big = ImageFont.load_default()

    look = [0.0, 0.0, nominal_height(p) * 0.55 + 0.02]
    dist = 1.9 * (nominal_height(p) + p["core_height"])
    J, G = (lambda n: joint_pos(model, data, n)), (lambda n: geom_pos(model, data, n))

    # ---- side view (looking from the left, +x to the right) -------------------
    rgb, seg = render(model, data, 90, 0, dist, look)
    side = Image.new("RGB", (PW, PH), "white"); side.paste(Image.fromarray(rgb), (PX, PY))
    proj0 = fit_projection(model, data, seg)
    proj = lambda q: (proj0(q)[0] + PX, proj0(q)[1] + PY)
    hip_cluster = J("hip_pitch_l")
    items = [
        (hip_cluster, ["HIP CLUSTER (left leg shown)", "hip yaw, thigh and shin servos", "all three inside the hip box"], HIP, (-200, 0)),
        (G("coupler_l"), ["parallelogram coupler", "shin servo -> knee"], HIP, (-220, 70)),
        (J("knee_l"), ["KNEE joint", "set by the shin servo through", "the parallelogram (no motor here)"], HIP, (230, -60)),
        (J("ankle_l"), ["ANKLE: passive hinge + spring", "no servo; rests toe-up, flattens under load"], PASSIVE, (230, 10)),
        (G("tread_drive_l"), ["TREAD DRIVE on the shin", "small BLDC + self-locking worm,", "belt down to the ankle pulley"], SHIN, (-220, 130)),
        (G("foot_l_r1"), ["tread rollers under the sole", "(driven together, locked when stopped)"], PASSIVE, (210, 80)),
        (G("payload"), ["battery (inside core)"], PASSIVE, (-230, -110)),
        (G("hip") , ["hip box"], PASSIVE, (240, -140)),
    ]
    annotate(side, proj, items, font)
    ImageDraw.Draw(side).text((16, 12), "SIDE VIEW  (forward = right)", fill=(0, 0, 0), font=big)

    # ---- front view (looking from the front, +y to the right) -----------------
    rgb, seg = render(model, data, 180, 0, dist, look)
    front = Image.new("RGB", (PW, PH), "white"); front.paste(Image.fromarray(rgb), (PX, PY))
    proj0 = fit_projection(model, data, seg)
    proj = lambda q: (proj0(q)[0] + PX, proj0(q)[1] + PY)
    items = [
        (J("hip_yaw_l"), ["hip yaw servo (vertical axis)", "steering; +-45 deg"], HIP, (200, -80)),
        (J("hip_pitch_r"), ["thigh servo (sideways axis)", "fore / aft swing"], HIP, (-200, -50)),
        (J("knee_r"), ["knee: shin servo via", "parallelogram (up / down)"], HIP, (-190, 0)),
        (J("ankle_r"), ["ankle: passive, sprung"], PASSIVE, (-190, 30)),
        (G("tread_drive_l"), ["tread drive"], SHIN, (190, 10)),
        (G("foot_l_r1"), ["sole + rollers", f"{2*15} mm tread width"], PASSIVE, (190, 60)),
        (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fan_l")], ["fan on a single", "pitch servo"], FAN, (170, -70)),
        (G("core"), ["core (battery, electronics)"], PASSIVE, (-200, -100)),
    ]
    annotate(front, proj, items, font)
    ImageDraw.Draw(front).text((16, 12), "FRONT VIEW", fill=(0, 0, 0), font=big)

    # ---- sheet with legend ------------------------------------------------------
    sheet = Image.new("RGB", (2 * PW, PH + 110), "white")
    sheet.paste(side, (0, 0))
    sheet.paste(front, (PW, 0))
    d = ImageDraw.Draw(sheet)
    y = PH + 12
    d.text((16, y), "Actuators per leg as modelled (archetype B: 3 hip servos + tread drive, passive ankle):", fill=(0, 0, 0), font=big)
    legend = [
        (HIP, "red: driven by a metal-gear servo in the HIP cluster (yaw and thigh directly; knee through the parallelogram from the shin servo)"),
        (SHIN, "orange: tread drive motor + worm, the only motor below the hip (on the shin at the knee)"),
        (PASSIVE, "blue: structure / unpowered parts, incl. the sprung ankle"),
        (FAN, "purple: fans, each on one pitch servo (independent channels)"),
    ]
    for i, (col, txt) in enumerate(legend):
        yy = y + 30 + 18 * i
        d.rectangle([20, yy + 3, 34, yy + 15], fill=col)
        d.text((42, yy), txt, fill=(0, 0, 0), font=font)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
