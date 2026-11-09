r"""What the core's VL53L8CX sees while a walker walks (evol_mech/skater/lidar.py).

    .venv\Scripts\python tools\lidar_view.py                                  # best walker, flat floor
    .venv\Scripts\python tools\lidar_view.py --terrain cobblestone --seed 4 --duration 4

Writes to runs/lidar_view/<terrain>/:
  lidar.mp4     the walk from a chase camera with the 64 zone beams drawn to where they land (colour = the
                measured range, red = no valid return), and the 8 x 8 range image beside it as the part
                reports it (row 0 = far edge, column 0 = the robot's left)
  sheet.png     one frame: the scene, the range image with the numbers, and a top view of the field's
                footprint with the measured points against the true ground
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evol_mech.skater.spec import encode                 # noqa: E402
from evol_mech.skater.rollout import rollout             # noqa: E402
from evol_mech.skater.lidar import ZONES, STATUS_VALID   # noqa: E402

W, H, PANEL = 640, 480, 368
CMAP = matplotlib.colormaps["viridis_r"]


def range_colour(d, lo, hi):
    if not np.isfinite(d):
        return np.array([0.9, 0.1, 0.1, 0.9])
    c = np.array(CMAP(np.clip((d - lo) / (hi - lo), 0, 1)))
    c[3] = 0.9
    return c


def draw_beams(scene, L, lo, hi):
    """64 lines from the module to each zone's true centre-ray hit, coloured by the zone's reported range."""
    o = L.origin_world
    for r in range(ZONES):
        for c in range(ZONES):
            h = L.hits_world[r, c]
            if scene.ngeom >= scene.maxgeom or not np.all(np.isfinite(h)):
                continue
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9),
                                range_colour(L.dist[r, c], lo, hi).astype(np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 1.5, o, h)
            scene.ngeom += 1
            g = scene.geoms[scene.ngeom]              # a dot where the beam lands
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]), h, np.eye(3).ravel(),
                                range_colour(L.dist[r, c], lo, hi).astype(np.float32))
            scene.ngeom += 1


def range_panel(L, lo, hi, size=PANEL):
    img = Image.new("RGB", (size, size + 40), (250, 250, 250))
    dr = ImageDraw.Draw(img)
    cell = (size - 20) // ZONES
    for r in range(ZONES):
        for c in range(ZONES):
            d = L.dist[r, c]
            col = tuple(int(255 * v) for v in range_colour(d, lo, hi)[:3])
            x0, y0 = 10 + c * cell, 30 + r * cell
            dr.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=col)
            txt = f"{100 * d:.0f}" if np.isfinite(d) else "--"
            dr.text((x0 + cell // 2 - 7, y0 + cell // 2 - 5), txt, fill=(255, 255, 255) if np.isfinite(d) and d > (lo + hi) / 2 else (0, 0, 0))
    dr.text((10, 8), f"VL53L8CX 8x8 range, cm   t={L.frame_t:.2f}s   (far edge on top, robot's left on the left)", fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walker", default="phase1m_seed")
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--sheet_t", type=float, default=None, help="time of the sheet's frame (default: mid-run)")
    ap.add_argument("--max_range", type=float, default=None, help="lidar_max_range_m (1.1 = daylight, dark floor)")
    args = ap.parse_args()
    out = ROOT / "runs" / "lidar_view" / args.terrain
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(ROOT / "configs/skater_walk_phase1m.json"))
    g = encode(json.load(open(ROOT / f"configs/archetypes/skater_walker_{args.walker}.json"))["phys"])
    task = {k: v for k, v in cfg["tasks"][0].items() if k not in ("push", "mass_error", "friction_error", "repeats", "weight")}
    task.update(terrain=args.terrain, duration=args.duration, lidar=1)
    if args.max_range:
        task["lidar_max_range_m"] = args.max_range
    lo, hi = 0.30, 0.75
    sheet_t = args.sheet_t if args.sheet_t is not None else args.duration / 2

    st = {"r": None, "next": 0.0, "L": None, "sheet": None}
    frames = []
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.distance, cam.elevation, cam.azimuth = 1.3, -28, 140

    def on_lidar(L, t):
        st["L"] = L
        if st["sheet"] is None and t >= sheet_t:
            st["sheet"] = dict(t=L.frame_t, dist=L.dist.copy(), hits=L.hits_world.copy(), pts=L.points_world().copy(),
                               origin=L.origin_world.copy(), rot=L.rot_world.copy(), status=L.status.copy())
            st["sheet_frame"] = True

    def on_frame(m, d, t):
        if st["r"] is None:
            st["r"] = mujoco.Renderer(m, H, W)
            cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hip")
        if t + 1e-9 < st["next"] and not st.pop("sheet_frame", False):
            return
        st["r"].update_scene(d, cam)
        if st["L"] is not None:
            draw_beams(st["r"].scene, st["L"], lo, hi)
        img = st["r"].render().copy()
        if st["sheet"] is not None and "scene" not in st["sheet"]:
            st["sheet"]["scene"] = img
        if t + 1e-9 >= st["next"]:
            st["next"] += 1 / 30
            canvas = Image.new("RGB", (W + PANEL, H), (250, 250, 250))
            canvas.paste(Image.fromarray(img), (0, 0))
            if st["L"] is not None:
                canvas.paste(range_panel(st["L"], lo, hi), (W, 60))
            frames.append(np.asarray(canvas))

    res = rollout(g, task, args.seed, frame_cb=on_frame, lidar_cb=on_lidar, stop_on_fall=False)
    if st["r"] is not None:
        st["r"].close()
    imageio.mimwrite(str(out / "lidar.mp4"), frames, fps=30, codec="libx264",
                     pixelformat="yuv420p", output_params=["-crf", "28", "-preset", "slow", "-movflags", "+faststart"])

    s = st["sheet"]
    fig = plt.figure(figsize=(16, 5.6))
    ax0 = fig.add_axes([0.0, 0.02, 0.34, 0.9])
    ax0.imshow(s["scene"])
    ax0.set_axis_off()
    ax0.set_title(f"{args.walker} on {args.terrain}, t = {s['t']:.2f} s: the 64 zone beams")
    ax1 = fig.add_axes([0.36, 0.1, 0.25, 0.78])
    im = ax1.imshow(100 * s["dist"], cmap=CMAP, vmin=100 * lo, vmax=100 * hi)
    for r in range(ZONES):
        for c in range(ZONES):
            d = s["dist"][r, c]
            ax1.text(c, r, f"{100 * d:.0f}" if np.isfinite(d) else "--", ha="center", va="center", fontsize=8,
                     color="white" if np.isfinite(d) and d > (lo + hi) / 2 else "black")
    ax1.set_xticks(range(ZONES)); ax1.set_yticks(range(ZONES))
    ax1.set_xlabel("column (0 = robot's left)"); ax1.set_ylabel("row (0 = far edge)")
    ax1.set_title("range per zone, cm (8 x 8, 15 Hz)")
    fig.colorbar(im, ax=ax1, fraction=0.046)
    ax2 = fig.add_axes([0.69, 0.1, 0.27, 0.78])
    o, hits, pts = s["origin"], s["hits"], s["pts"]
    fwd = -s["rot"][:, 0]                         # the field's far direction (sensor -x), levelled: the robot's heading
    fwd = np.array([fwd[0], fwd[1], 0.0]) / np.hypot(fwd[0], fwd[1])
    left = np.array([-fwd[1], fwd[0], 0.0])
    to_robot = lambda q: ((q - o) @ fwd, (q - o) @ left)
    hx, hy = to_robot(hits)
    px, py = to_robot(pts)
    ax2.plot(hx.ravel(), hy.ravel(), "o", ms=7, mfc="none", mec="0.5", label="true centre-ray hit")
    sc = ax2.scatter(px, py, c=100 * pts[..., 2], cmap="coolwarm", s=22,
                     vmin=-4, vmax=10, label="measured point (height, cm)")
    for side in (1, -1):                          # the feet's track, for scale
        ax2.axhline(side * 0.05, color="0.85", lw=8, zorder=0)
    ax2.plot([0], [0], "k^", ms=9, label="module")
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
    ax2.set_xlabel("ahead of the module, m"); ax2.set_ylabel("to the robot's left, m")
    ax2.set_title("footprint, top view (grey bands: the feet's track)")
    fig.colorbar(sc, ax=ax2, fraction=0.046, label="measured height, cm")
    ax2.legend(loc="upper left", fontsize=7)
    fig.savefig(out / "sheet.png", dpi=110)
    print(json.dumps({k: res.get(k) for k in ("fitness", "time_upright", "fall_dir", "lidar_frames", "lidar_valid_frac", "stride_m")}, default=float))
    near, far = np.nanmin(hx), np.nanmax(hx)
    print(f"footprint: {near:.2f} to {far:.2f} m ahead of the module; width {np.nanmax(hy[0]) - np.nanmin(hy[0]):.2f} m at the far edge, "
          f"{np.nanmax(hy[-1]) - np.nanmin(hy[-1]):.2f} m at the near edge; module {o[2]:.3f} m up")
    print("wrote", out)


if __name__ == "__main__":
    main()
