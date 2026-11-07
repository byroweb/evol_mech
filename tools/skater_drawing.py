r"""Three-view engineering drawing of a skater archetype with the genes pointing at
the parts they size: front view, side view, isometric, plus a table of the gene
values. Run it on any archetype or GA best to see what a body is.

    .venv\Scripts\python tools\skater_drawing.py configs\archetypes\skater_walker_panto5_seed.json
    .venv\Scripts\python tools\skater_drawing.py runs\skater_walk_panto5\best.json --out docs\skater_drawing.png

Pixel positions come from a segmentation render fitted to the geoms' world
positions (as tools/skater_layout.py does), so labels land on the parts in any
view. Pantograph bodies (`leg_drive` 1) get the pantograph's genes, four-bar
bodies the thigh / shin / coupler genes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evol_mech.data.store import load_genome
from evol_mech.skater import DEFAULT_NORM, make_model, make_data, nominal_height, decode, encode
from evol_mech.skater.structure import is_pantograph, roll_ratio, roll_arm, horn_stiffness, leg_reach, panto_reach

W, H = 640, 480                  # the default offscreen framebuffer
PX, PY = 320, 40                  # label margins around each render
PW, PH = W + 2 * PX, H + 2 * PY
GEOM = (30, 30, 30)
LEG = (190, 40, 40)               # leg linkage genes
BODY = (30, 90, 200)              # body / core genes
FAN = (120, 60, 160)
FOOT = (170, 110, 0)
DRIVE = (0, 130, 90)              # servo / stiffness genes


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
    return rgb, seg[:, :, 0].astype(int)


def fit_projection(model, data, seg):
    """Least-squares affine world -> pixel from the visible geoms' centroids."""
    pts, pix = [], []
    for g in range(model.ngeom):
        ys, xs = np.nonzero(seg == g)
        if len(xs) < 20:
            continue
        pts.append(data.geom_xpos[g])
        pix.append((xs.mean(), ys.mean()))
    A = np.hstack([np.array(pts), np.ones((len(pts), 1))])
    M, *_ = np.linalg.lstsq(A, np.array(pix), rcond=None)
    return lambda q: tuple((np.append(q, 1.0) @ M).tolist())


class Scene:
    def __init__(self, model, data):
        self.model, self.data = model, data

    def body(self, name):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    def geom(self, name):
        return self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)].copy()

    def joint(self, name):
        return self.data.xanchor[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)].copy()

    def mid(self, a, b):
        return (a + b) / 2


def labels(sc: Scene, p: dict) -> list:
    """(world point, lines, colour) for every gene that sizes a visible part."""
    s = "l"
    hip = sc.geom("hip")
    core = sc.geom("core")
    fan = sc.body("fan_l")
    sole = sc.geom(f"foot_{s}_sole")
    items = [
        (hip, [f"hip_width {p['hip_width'] * 100:.1f} cm", "distance between the leg planes"], BODY),
        (core, [f"core_height {p['core_height'] * 100:.0f} cm", f"battery_mah {p['battery_mah']:.0f}"], BODY),
        (sc.geom("payload"), [f"payload_x {p['payload_x'] * 100:+.0f} cm, payload_zf {p['payload_zf']:+.2f}", "battery position in the core"], BODY),
        (fan, [f"fan_diam_in {p['fan_diam_in']:.0f}\", fan_x {p['fan_x'] * 100:+.0f} cm", f"fan_zf {p['fan_zf']:.2f}, fan_gap {p['fan_gap'] * 100:.0f} cm", "one pitch servo per fan"], FAN),
        (sole, [f"sole_len {p['sole_len'] * 100:.0f} cm, tread_w {p['tread_w'] * 100:.0f} cm", f"foot_type {p['foot_type']:.0f}: tread sole, passive sprung ankle"], FOOT),
        (sc.geom(f"tread_drive_{s}"), [f"tread_motor_w {p['tread_motor_w']:.0f} W", f"worm_ratio {p['worm_ratio']:.0f}, pulley_radius {p['pulley_radius'] * 1000:.0f} mm"], FOOT),
    ]
    if is_pantograph(p):
        hf, hr = sc.body(f"horn_f_{s}"), sc.body(f"horn_r_{s}")
        rod_f, rod_r = sc.body(f"rod_f_{s}"), sc.body(f"rod_r_{s}")
        link_b, link_c = sc.body(f"link_b_{s}"), sc.body(f"link_c_{s}")
        foot = sc.body(f"foot_{s}")
        roll_horn, roll_link = sc.body(f"roll_horn_{s}"), sc.body(f"roll_link_{s}")
        servo0 = sc.geom(f"servo_{s}0")
        bracket = sc.geom(f"bracket_{s}")
        items += [
            (sc.mid(hf, hr), [f"servo_dx {p['servo_dx'] * 100:.1f} cm", "fore-aft spacing of the two leg servos", f"servo_kgcm {p['servo_kgcm']:.0f} (all five servos one part)"], DRIVE),
            (sc.mid(hf, rod_f), [f"horn_len {p['horn_len'] * 1000:.0f} mm", f"horn_k_log10 {p['horn_k_log10']:.2f} = {horn_stiffness(p):.0f} N m/rad", "every horn: one part, flexes in its sweep plane"], LEG),
            (sc.mid(rod_f, rod_r), [f"rod_len {p['rod_len'] * 100:.1f} cm", "horn tip to the crossing pin (knee node)"], LEG),
            (sc.mid(link_b, link_c), [f"rod_ext {p['rod_ext'] * 100:.1f} cm", "rods continued past the pin and the", "parallelogram's short sides to the ankle node"], LEG),
            (sc.mid(link_c, foot), [f"strut_len {p['strut_len'] * 100:.1f} cm", "ankle node down to the sprung foot"], LEG),
            (sc.mid(rod_f, link_c), [f"crouch {p['crouch']:.2f} rad", f"parked leg: reach {panto_reach(p) * 100:.0f} cm x cos(crouch/2)", f"= {nominal_height(p) * 100:.0f} cm hip height"], LEG),
            (bracket, [f"roll lever = bracket {roll_arm(p) * 100:.1f} cm / horn", f"-> roll ratio {roll_ratio(p):.2f} (derived, not a gene)", "pivot at the hip centre, both leg servos ride on it"], DRIVE),
            (sc.mid(roll_horn, roll_link), ["roll servo horn + link", "(horn_len, horn_k_log10 again)"], DRIVE),
        ]
    else:
        items += [
            (sc.mid(sc.joint(f"hip_pitch_{s}"), sc.joint(f"knee_{s}")), [f"thigh_len {p['thigh_len'] * 100:.1f} cm", "thigh servo direct"], LEG),
            (sc.mid(sc.joint(f"knee_{s}"), sc.joint(f"ankle_{s}")), [f"shin_len {p['shin_len'] * 100:.1f} cm", f"crouch {p['crouch']:.2f} rad"], LEG),
            (sc.geom(f"coupler_{s}"), ["four-bar coupler from the shin servo", f"roll_ratio {p['roll_ratio']:.2f} (hip roll lever)"], DRIVE),
        ]
    return items


def annotate(img, proj, items, font):
    """Labels in two columns, left and right of the render (each part's nearer side),
    top to bottom in the order of the parts, with leaders to the parts."""
    d = ImageDraw.Draw(img)
    pts = [proj(q) for q, _, _ in items]
    # each part's nearer side, then the columns balanced by moving the parts nearest
    # the centre line across, so neither column has to squeeze its labels
    cols = {"left": [i for i in range(len(items)) if pts[i][0] < PW / 2], "right": [i for i in range(len(items)) if pts[i][0] >= PW / 2]}
    half = (len(items) + 1) // 2
    for big_side, small_side in (("left", "right"), ("right", "left")):
        while len(cols[big_side]) > half:
            move = min(cols[big_side], key=lambda i: abs(pts[i][0] - PW / 2))
            cols[big_side].remove(move)
            cols[small_side].append(move)
    for side in ("left", "right"):
        col_items = sorted(cols[side], key=lambda i: pts[i][1])
        n = len(col_items)
        if n == 0:
            continue
        step = min(3 * 16 + 10, (PH - 60) // n)
        y0 = max(40, (PH - step * n) // 2)
        for k, i in enumerate(col_items):
            (x, y), (q, lines, col) = pts[i], items[i]
            ly = y0 + step * k + step / 2
            tw = max(d.textlength(t, font=font) for t in lines)
            th = 16 * len(lines)
            if side == "left":
                bx = 12
                ax = bx + tw + 6
            else:
                bx = PW - 12 - tw
                ax = bx - 6
            d.line([(x, y), (ax, ly)], fill=col, width=2)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col, width=3)
            d.rectangle([bx - 4, ly - th / 2 - 3, bx + tw + 4, ly + th / 2 + 3], fill=(255, 255, 255), outline=col)
            for j, t in enumerate(lines):
                d.text((bx, ly - th / 2 + 16 * j), t, fill=col if j == 0 else GEOM, font=font)


def view(model, data, p, azimuth, elevation, dist, look, title, font, big):
    rgb, seg = render(model, data, azimuth, elevation, dist, look)
    img = Image.new("RGB", (PW, PH), "white")
    img.paste(Image.fromarray(rgb), (PX, PY))
    proj0 = fit_projection(model, data, seg)
    proj = lambda q: (proj0(q)[0] + PX, proj0(q)[1] + PY)
    annotate(img, proj, labels(Scene(model, data), p), font)
    ImageDraw.Draw(img).text((PX, 14), title, fill=(0, 0, 0), font=big)
    return img


def gene_table(p: dict, font, big, width: int, note: str) -> Image.Image:
    from evol_mech.skater.spec import PARAMS
    groups: dict[str, list[str]] = {}
    for q in PARAMS:
        groups.setdefault(q.group, []).append(f"{q.name} {p[q.name]:.3g}")
    rows = []
    for g, v in groups.items():                       # wrap each group's row to the sheet width
        line = f"{g}: "
        for item in v:
            if len(line) + len(item) > 230:
                rows.append(line.rstrip(", "))
                line = " " * (len(g) + 2)
            line += item + ", "
        rows.append(line.rstrip(", "))
    img = Image.new("RGB", (width, 40 + 18 * (len(rows) + 2)), "white")
    d = ImageDraw.Draw(img)
    d.text((16, 8), note, fill=(0, 0, 0), font=big)
    for i, r in enumerate(rows):
        d.text((16, 40 + 18 * i), r, fill=GEOM, font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best", nargs="?", help="archetype json (phys) or a GA best.json; default body otherwise")
    ap.add_argument("--out", default="runs/skater_drawing.png")
    args = ap.parse_args()
    if args.best and "phys" in json.load(open(args.best)):
        genome = encode(json.load(open(args.best))["phys"])
    elif args.best:
        genome = load_genome(args.best)
    else:
        genome = DEFAULT_NORM
    model, p = make_model(genome, "flat", np.random.default_rng(0))
    data = make_data(model, p)
    mujoco.mj_forward(model, data)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        big = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = big = ImageFont.load_default()

    look = [0.0, 0.0, nominal_height(p) * 0.55 + 0.04]
    dist = 1.6 * (nominal_height(p) + p["core_height"])
    front = view(model, data, p, 180, 0, dist, look, "FRONT VIEW", font, big)
    side = view(model, data, p, 90, 0, dist, look, "SIDE VIEW  (forward = right)", font, big)
    iso = view(model, data, p, 135, -28, dist * 1.15, look, "ISOMETRIC  (from the front-left, above)", font, big)

    drive = "pantograph leg (leg_drive 1)" if is_pantograph(p) else "four-bar leg (leg_drive 0)"
    name = Path(args.best).stem if args.best else "default body"
    note = (f"{name}: {drive}, hip {p['hip_width'] * 100:.0f} cm, leg reach {leg_reach(p) * 100:.0f} cm, "
            f"hip height {nominal_height(p) * 100:.0f} cm. Gene values (red leg, green drive, blue body, purple fan, brown foot):")
    table = gene_table(p, font, big, 2 * PW, note)
    sheet = Image.new("RGB", (2 * PW, 2 * PH + table.height), "white")
    sheet.paste(front, (0, 0))
    sheet.paste(side, (PW, 0))
    sheet.paste(iso, (0, PH))
    sheet.paste(table, (0, 2 * PH))
    d = ImageDraw.Draw(sheet)
    d.text((PW + 40, PH + 40), "Legend", fill=(0, 0, 0), font=big)
    for i, (col, txt) in enumerate(((LEG, "leg linkage lengths and the parked crouch"), (DRIVE, "servo spacing, horns, stiffness and the roll lever"),
                                    (BODY, "hip, core and battery"), (FAN, "fans"), (FOOT, "sole and tread drive"))):
        d.rectangle([PW + 40, PH + 80 + 22 * i, PW + 54, PH + 94 + 22 * i], fill=col)
        d.text((PW + 62, PH + 78 + 22 * i), txt, fill=GEOM, font=font)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
