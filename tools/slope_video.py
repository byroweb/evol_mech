r"""Watch a walker through a slope's start: two chase views (side and rear) at reduced speed with a live readout.

    .venv\Scripts\python tools\slope_video.py --deg 6 --head 90 --seed 104          # 6 deg rising to the left
    .venv\Scripts\python tools\slope_video.py --deg 10 --head 0 --blend 1.5 --slowmo 0.5

Writes runs/slope_video/<name>.mp4. The readout: time, hip x (the grade starts at slope_x0 and is full
slope_blend later), body pitch (+ nose down), roll (+ right side down), heading, each foot's centre of pressure
as a bar from heel to toe (red = pinned at an edge), the slope estimate against the true grade under the hip.
The walker is phase1m_seed with the slope logic on (lidar map + footfall check, the gene defaults), unless
--genome or --off.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evol_mech.skater.spec import encode                       # noqa: E402
from evol_mech.skater import controller as C                    # noqa: E402
from evol_mech.skater.rollout import rollout                    # noqa: E402
from evol_mech.sim.terrain import slope_height, SLOPE_X0, SLOPE_BLEND   # noqa: E402

VW, VH, PH = 480, 400, 144          # each view, the readout panel's height (960 x 544 overall)
FPS = 30


class _Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, default=6.0)
    ap.add_argument("--head", type=float, default=90.0, help="0 uphill ahead, 180 downhill, 90 rising left, -90 right")
    ap.add_argument("--blend", type=float, default=SLOPE_BLEND)
    ap.add_argument("--x0", type=float, default=SLOPE_X0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--slowmo", type=float, default=0.5, help="playback speed (0.5 = half speed)")
    ap.add_argument("--genome", help="best.json of a run (default phase1m_seed)")
    ap.add_argument("--off", action="store_true", help="slope logic off")
    ap.add_argument("--name")
    ap.add_argument("--extra", default="{}", help="JSON of extra task keys")
    args = ap.parse_args()

    if args.genome:
        g = np.array(json.load(open(args.genome))["genome"], dtype=float)
    else:
        g = encode(json.load(open(ROOT / "configs/archetypes/skater_walker_phase1m_seed.json"))["phys"])
    cfg = json.load(open(ROOT / "configs/skater_walk_phase1m.json"))
    task = {k: v for k, v in cfg["tasks"][0].items()
            if k not in ("push", "mass_error", "friction_error", "repeats", "weight", "aggregate")}
    task.update(terrain="slope", slope_deg=args.deg, slope_heading_deg=args.head, slope_blend=args.blend,
                slope_x0=args.x0, duration=args.duration, height_ref="feet", lidar=1)
    if not args.off:
        task["slope_est"] = "map"
    task.update(json.loads(args.extra))
    name = args.name or f"{'off_' if args.off else ''}{args.deg:g}deg_head{args.head:g}_blend{args.blend:g}_seed{args.seed}"
    out = ROOT / "runs" / "slope_video"
    out.mkdir(parents=True, exist_ok=True)

    st = {"ctrl": None, "r": None, "next": 0.0, "fall_t": None}
    orig = C.SkaterController.act

    def act(self, data, t):
        st["ctrl"] = self
        return orig(self, data, t)
    C.SkaterController.act = act

    cams = []
    for az, el in ((90, -8), (180, -12)):           # side (from the robot's left), rear
        c = mujoco.MjvCamera()
        c.type, c.distance, c.elevation, c.azimuth = mujoco.mjtCamera.mjCAMERA_TRACKING, 1.1, el, az
        cams.append(c)
    frames = []

    def panel(data, t):
        img = Image.new("RGB", (2 * VW, PH), (250, 250, 250))
        dr = ImageDraw.Draw(img)
        ctrl = st["ctrl"]
        q = data.qpos
        roll, pitch = C.quat_to_roll_pitch(q[3:7])
        yaw = C.quat_to_yaw(q[3:7])
        x, y = float(q[0]), float(q[1])
        e = 0.01
        sh = lambda u, v: float(slope_height(u, v, args.deg, args.head, args.x0, args.blend))
        gx, gy = (sh(x + e, y) - sh(x - e, y)) / (2 * e), (sh(x, y + e) - sh(x, y - e)) / (2 * e)
        cy, sy = math.cos(yaw), math.sin(yaw)
        true = np.degrees((math.atan(cy * gx + sy * gy), math.atan(-sy * gx + cy * gy)))
        est = np.degrees(ctrl.grade) if ctrl is not None else (0.0, 0.0)
        u = min(1.0, max(0.0, (x - args.x0) / args.blend))
        zone = "flat" if x < args.x0 else ("TRANSITION" if u < 1 else "on the grade")
        col = (200, 60, 0) if zone == "TRANSITION" else (0, 0, 0)
        dr.text((10, 8), f"t {t:5.2f} s   x {x:5.2f} m  ({zone})   playback x{args.slowmo:g}", fill=col)
        dr.text((10, 28), f"pitch {math.degrees(pitch):+6.1f} deg (+ nose down)   roll {math.degrees(roll):+6.1f} "
                          f"(+ right down)   heading {math.degrees(yaw):+6.1f}", fill=(0, 0, 0))
        dr.text((10, 48), f"grade along: est {est[0]:+5.1f}  true {true[0]:+5.1f}     across: est {est[1]:+5.1f}  "
                          f"true {true[1]:+5.1f}  deg", fill=(0, 0, 0))
        dr.text((10, 68), f"{args.deg:g} deg slope, uphill heading {args.head:g} deg, transition {args.blend:g} m, "
                          f"seed {args.seed}{', slope logic OFF' if args.off else ''}", fill=(90, 90, 90))
        if st["fall_t"] is not None:
            dr.text((10, 88), f"FELL at {st['fall_t']:.2f} s", fill=(200, 0, 0))
        if ctrl is not None:                        # centre-of-pressure bars, heel (left end) to toe (right end)
            for k, s in enumerate("lr"):
                x0, y0 = 560, 20 + 56 * k
                dr.text((x0 - 70, y0 + 5), f"{'left' if s == 'l' else 'right'} foot", fill=(0, 0, 0))
                dr.rectangle([x0, y0, x0 + 360, y0 + 22], outline=(120, 120, 120))
                dr.text((x0 + 2, y0 + 26), "heel", fill=(90, 90, 90))
                dr.text((x0 + 336, y0 + 26), "toe", fill=(90, 90, 90))
                sens = ctrl.sensors
                if sens.contact[s]:
                    c = max(-1.0, min(1.0, sens.cop[s] / max(1e-6, sens.pad_x)))
                    cx = x0 + 180 + 180 * c
                    shade = (220, 40, 40) if abs(c) > 0.8 else (40, 120, 220)
                    dr.rectangle([cx - 6, y0 + 2, cx + 6, y0 + 20], fill=shade)
                    dr.text((x0 + 150, y0 + 5), f"load {sens.load[s]:.2f}", fill=(0, 0, 0))
                else:
                    dr.text((x0 + 150, y0 + 5), "in the air", fill=(150, 150, 150))
        return img

    def on_frame(m, d, t):
        if st["r"] is None:
            st["r"] = mujoco.Renderer(m, VH, VW)
            for c in cams:
                c.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hip")
        if st["fall_t"] is None:
            roll, pitch = C.quat_to_roll_pitch(d.qpos[3:7])
            if max(abs(roll), abs(pitch)) > math.radians(task.get("fall_pitch_deg", 20)):
                st["fall_t"] = t
        if t + 1e-9 < st["next"]:
            return
        st["next"] += args.slowmo / FPS
        views = []
        for c in cams:
            st["r"].update_scene(d, c)
            views.append(st["r"].render().copy())
        canvas = Image.new("RGB", (2 * VW, VH + PH), (250, 250, 250))
        canvas.paste(Image.fromarray(views[0]), (0, 0))
        canvas.paste(Image.fromarray(views[1]), (VW, 0))
        canvas.paste(panel(d, t), (0, VH))
        frames.append(np.asarray(canvas))
        if st["fall_t"] is not None and t > st["fall_t"] + 1.0:
            raise _Stop

    try:
        rollout(g, task, args.seed, frame_cb=on_frame, stop_on_fall=False)
    except _Stop:
        pass
    if st["r"] is not None:
        st["r"].close()
    path = out / f"{name}.mp4"
    imageio.mimwrite(str(path), frames, fps=FPS, codec="libx264", pixelformat="yuv420p",
                     output_params=["-crf", "28", "-preset", "slow", "-movflags", "+faststart"])
    print(f"wrote {path}  ({len(frames)} frames)  fell at {st['fall_t']}")


if __name__ == "__main__":
    main()
