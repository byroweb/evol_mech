r"""Population video of a GA run: every individual of each generation plays in a
tile grid, the generation's best is outlined in gold, and between generations
branches are drawn from that best to its children (elite copy, mutants,
crossovers) in the next generation's grid, which then slides up to play.

    .venv\Scripts\python tools\evolution_video.py runs\skater_lineage                 # first task, -> runs/skater_lineage/videos/evolution_<task>.mp4
    .venv\Scripts\python tools\evolution_video.py runs\skater_lineage --task slalom --clip 4 --gens 6

Needs the per-generation `parents` / `kinds` / `gen_seed` arrays that the GA
has stored since the lineage change; older runs render without branches.
Rendering every individual is the slow part: pop x gens rollouts at tile size.
"""
from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evol_mech.bodies import get_body
from evol_mech.evolve.ga import KIND_ELITE, KIND_MUTANT, KIND_CROSSOVER, KIND_NAMES

FPS = 20
LABEL_H = 18            # fitness strip under each tile
HEADER_H = 66           # title line + legend line
BAND_PAD = 28           # above and below the best tile's copy in the band between grids
BORDER = 4

WHITE = (255, 255, 255)
BG = (24, 26, 30)
PANEL = (40, 43, 48)
TEXT = (230, 230, 230)
DIM = (120, 120, 120)
GOLD = (255, 196, 0)
ELITE = (90, 170, 255)          # other carried-over elites
CHILD = (80, 220, 120)          # children of the previous best
PARENT_OTHER = (150, 110, 220)  # children of another parent (crossover partner)
GREY = (70, 70, 70)


# ---- rendering one individual --------------------------------------------------

def _render_tile(args):
    genome, task, seed, body_name, fans_enabled, w, h, clip_s = args
    frames: list[np.ndarray] = []
    state = {"r": None, "next_t": 0.0}
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.distance, cam.elevation, cam.azimuth = 1.1, -15, 135

    def cb(model, data, t):
        if state["r"] is None:
            state["r"] = mujoco.Renderer(model, h, w)
            cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
        if t >= state["next_t"]:
            state["r"].update_scene(data, cam)
            frames.append(state["r"].render().copy())
            state["next_t"] += 1.0 / FPS

    body = get_body(body_name)
    try:
        res = body.rollout(genome, dict(task, duration=clip_s), seed=seed, fans_enabled=fans_enabled, frame_cb=cb)
        fell = bool(res.get("fell", False))
    except Exception:   # invalid body: blank tile
        fell = True
    if state["r"] is not None:
        state["r"].close()
    n_target = int(round(clip_s * FPS))
    if not frames:
        frames = [np.full((h, w, 3), 60, dtype=np.uint8)]
    while len(frames) < n_target:          # fell early: hold the last frame
        frames.append(frames[-1])
    return np.stack(frames[:n_target]), fell


# ---- grid geometry ---------------------------------------------------------------

class Grid:
    def __init__(self, n: int, tw: int, th: int, pad: int = 10):
        self.n, self.tw, self.th, self.pad = n, tw, th, pad
        self.cols = int(math.ceil(math.sqrt(n * 1.4)))
        self.rows = int(math.ceil(n / self.cols))
        self.cell_w = tw + pad
        self.cell_h = th + LABEL_H + pad
        self.w = self.cols * self.cell_w + pad
        self.h = self.rows * self.cell_h + pad

    def box(self, i: int, ox: int, oy: int) -> tuple[int, int, int, int]:
        r, c = divmod(i, self.cols)
        x = ox + self.pad + c * self.cell_w
        y = oy + self.pad + r * self.cell_h
        return x, y, x + self.tw, y + self.th

    def top_center(self, i, ox, oy):
        x0, y0, x1, _ = self.box(i, ox, oy)
        return ((x0 + x1) // 2, y0 - BORDER)

    def bottom_center(self, i, ox, oy):
        x0, _, x1, y1 = self.box(i, ox, oy)
        return ((x0 + x1) // 2, y1 + LABEL_H + BORDER)


# ---- drawing ---------------------------------------------------------------------

def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_grid(img: Image.Image, grid: Grid, ox: int, oy: int, tiles, fit, borders, labels, font, dim=1.0):
    """tiles: list of HxWx3 arrays (or None). borders: list of (colour, width) or None."""
    d = ImageDraw.Draw(img)
    for i in range(grid.n):
        x0, y0, x1, y1 = grid.box(i, ox, oy)
        tile = tiles[i]
        if tile is None:
            d.rectangle([x0, y0, x1, y1], fill=PANEL)
        else:
            t = Image.fromarray(tile)
            if dim < 1.0:
                t = Image.blend(Image.new("RGB", t.size, BG), t, dim)
            img.paste(t, (x0, y0))
        d.rectangle([x0, y1, x1, y1 + LABEL_H], fill=PANEL)
        d.text((x0 + 4, y1 + 2), labels[i], fill=TEXT if dim == 1.0 else DIM, font=font)
        b = borders[i]
        if b is not None:
            col, wd = b
            d.rectangle([x0 - wd, y0 - wd, x1 + wd, y1 + LABEL_H + wd], outline=col, width=wd)


def bezier(p0, p1, steps=24):
    (x0, y0), (x1, y1) = p0, p1
    ym = (y0 + y1) / 2
    pts = []
    for k in range(steps + 1):
        t = k / steps
        # cubic with vertical tangents at both ends
        x = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * x0 + 3 * (1 - t) * t ** 2 * x1 + t ** 3 * x1
        y = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * ym + 3 * (1 - t) * t ** 2 * ym + t ** 3 * y1
        pts.append((x, y))
    return pts


def draw_branches(d: ImageDraw.ImageDraw, src, dsts, progress: float):
    """dsts: list of ((x, y), colour, width). progress in [0, 1] animates the growth."""
    for dst, col, wd in dsts:
        pts = bezier(src, dst)
        k = max(2, int(round(progress * len(pts))))
        d.line(pts[:k], fill=col, width=wd, joint="curve")
        if progress >= 1.0:
            x, y = dst
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)


# ---- main ------------------------------------------------------------------------

def load_generations(run: Path, max_gens: int | None):
    files = sorted((run / "population").glob("gen_*.npz"))
    if max_gens:
        files = files[:max_gens]
    gens = []
    for f in files:
        z = np.load(f)
        gens.append({"genomes": z["genomes"], "fitness": z["fitness"],
                     "parents": z["parents"] if "parents" in z else np.full((len(z["fitness"]), 2), -1),
                     "kinds": z["kinds"] if "kinds" in z else np.zeros(len(z["fitness"]), dtype=int),
                     "gen_seed": int(z["gen_seed"]) if "gen_seed" in z else 0})
    return gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", help="task name from the run config (default: first)")
    ap.add_argument("--gens", type=int, help="only the first N generations")
    ap.add_argument("--clip", type=float, default=3.0, help="seconds of each individual's rollout to show")
    ap.add_argument("--tile", type=int, default=176, help="tile width in px (4:3)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.load(open(run / "config.json"))
    body_name = cfg.get("body", "biped")
    tasks = {t["name"]: t for t in cfg["tasks"]}
    task = tasks[args.task] if args.task else cfg["tasks"][0]
    gens = load_generations(run, args.gens)
    if not gens:
        raise SystemExit(f"no population files under {run}")
    n = len(gens[0]["fitness"])
    tw, th = args.tile, args.tile * 3 // 4
    grid = Grid(n, tw, th)
    out = Path(args.out) if args.out else run / "videos" / f"evolution_{task['name']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- render every individual --------------------------------------------
    jobs = [(g["genomes"][i], task, g["gen_seed"], body_name, cfg.get("fans_enabled", True), tw, th, args.clip)
            for g in gens for i in range(n)]
    print(f"rendering {len(jobs)} tiles ({len(gens)} generations x {n}) at {tw}x{th}, {args.clip:.1f} s each")
    if args.workers > 1:
        with Pool(args.workers) as pool:
            rendered = pool.map(_render_tile, jobs, chunksize=2)
    else:
        rendered = [_render_tile(j) for j in jobs]
    clips = [[rendered[g * n + i][0] for i in range(n)] for g in range(len(gens))]
    n_clip = clips[0][0].shape[0]

    # ---- compose --------------------------------------------------------------
    W = grid.w
    gap_h = th + LABEL_H + 2 * BAND_PAD          # band that holds the copy of the best between grids
    H = HEADER_H + grid.h + gap_h + grid.h + 10
    font, big, small = _font(13), _font(20, True), _font(12)
    top_y = HEADER_H
    band_y = HEADER_H + grid.h + BAND_PAD        # top of the best tile's copy
    copy_x = (W - tw) // 2
    bot_y = HEADER_H + grid.h + gap_h
    writer = imageio.get_writer(str(out), fps=FPS, codec="libx264", quality=7, macro_block_size=1)

    def labels_for(g, with_fitness=True):
        gen = gens[g]
        out_l = []
        for i in range(n):
            k = int(gen["kinds"][i])
            tag = {KIND_ELITE: "elite", KIND_MUTANT: "mutant", KIND_CROSSOVER: "cross"}.get(k, "")
            out_l.append(f"#{i:<2d} {gen['fitness'][i]:+.3f} {tag}" if with_fitness else f"#{i:<2d} {tag}")
        return out_l

    def borders_for(g, i_best, prev_best):
        gen = gens[g]
        b = []
        for i in range(n):
            par = gen["parents"][i]
            if i == i_best:
                b.append((GOLD, BORDER))
            elif prev_best is not None and prev_best in par:
                b.append((CHILD, 2))
            elif int(gen["kinds"][i]) == KIND_ELITE:
                b.append((ELITE, 2))
            else:
                b.append((GREY, 1))
        return b

    def header(d, g, i_best):
        gen = gens[g]
        d.rectangle([0, 0, W, HEADER_H], fill=BG)
        d.text((12, 8), f"generation {g}", fill=WHITE, font=big)
        d.text((190, 12), f"best #{i_best}  fitness {gen['fitness'][i_best]:+.3f}    mean {gen['fitness'].mean():+.3f}"
               f"    task {task['name']}  ({body_name}, {len(gens)} generations x {n})", fill=TEXT, font=font)
        legend = [(GOLD, "best of this generation"), (CHILD, "child of previous best"), (ELITE, "elite copy"),
                  (PARENT_OTHER, "crossover partner")]
        x = 12
        for col, txt in legend:
            d.rectangle([x, 44, x + 10, 54], fill=col)
            x += 16
            d.text((x, 41), txt, fill=TEXT, font=small)
            x += d.textlength(txt, font=small) + 18

    prev_best = None
    for g, gen in enumerate(gens):
        i_best = int(np.argmax(gen["fitness"]))
        labels = labels_for(g)
        borders = borders_for(g, i_best, prev_best)
        has_next = g + 1 < len(gens)
        if has_next:
            nxt = gens[g + 1]
            n_labels = labels_for(g + 1)
            n_blank = labels_for(g + 1, with_fitness=False)
            n_borders = borders_for(g + 1, int(np.argmax(nxt["fitness"])), i_best)
            # branches: best -> its children; its crossover partners in purple
            dsts, partner_dsts = [], []
            for i in range(n):
                pa, pb = (int(v) for v in nxt["parents"][i])
                if i_best in (pa, pb):
                    kind = int(nxt["kinds"][i])
                    col = ELITE if kind == KIND_ELITE else CHILD
                    dsts.append((grid.top_center(i, 0, bot_y), col, 3 if kind != KIND_ELITE else 4))
                    other = pb if pa == i_best else pa
                    if kind == KIND_CROSSOVER and other >= 0 and other != i_best:
                        partner_dsts.append((grid.bottom_center(other, 0, top_y), grid.top_center(i, 0, bot_y)))

        # phase A: play the generation, next generation's slots empty
        for f in range(n_clip):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            header(d, g, i_best)
            draw_grid(img, grid, 0, top_y, [clips[g][i][f] for i in range(n)], gen["fitness"], borders, labels, font)
            d.text((12, band_y + th // 2), f"generation {g + 1}" if has_next else "end of run", fill=DIM, font=font)
            if has_next:
                draw_grid(img, grid, 0, bot_y, [None] * n, None, [None] * n, [""] * n, font)
            writer.append_data(np.asarray(img))
        if not has_next:
            for _ in range(FPS):   # hold the final generation
                writer.append_data(np.asarray(img))
            break

        # phase B: freeze, reveal the next generation, grow the branches from the best
        last = [clips[g][i][-1] for i in range(n)]
        mid = [clips[g + 1][i][n_clip // 3] for i in range(n)]
        n_b = int(1.6 * FPS)
        for f in range(n_b):
            prog = min(1.0, f / (0.7 * n_b))
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            header(d, g, i_best)
            dim_borders = [(GOLD, 2) if i == i_best else (GREY, 1) for i in range(n)]
            draw_grid(img, grid, 0, top_y, last, gen["fitness"], dim_borders, labels, font, dim=0.35)
            draw_grid(img, grid, 0, bot_y, mid, nxt["fitness"], n_borders, n_blank, font, dim=0.8)
            # the best, copied into the band; its children fan out from the copy
            copy_box = (copy_x, band_y, copy_x + tw, band_y + th)
            d.line([grid.bottom_center(i_best, 0, top_y), (copy_x + tw // 2, band_y - BORDER)], fill=GOLD, width=2)
            paste_tile(img, copy_box, last[i_best], (GOLD, BORDER), f"best #{i_best}  {gen['fitness'][i_best]:+.3f}", font)
            d.text((12, band_y + th // 2 - 8), f"generation {g + 1}:", fill=TEXT, font=font)
            d.text((12, band_y + th // 2 + 8), f"children of #{i_best} branch from it", fill=TEXT, font=font)
            for (src, dst) in partner_dsts:
                draw_branches(d, src, [(dst, PARENT_OTHER, 1)], prog)
            draw_branches(d, (copy_x + tw // 2, band_y + th + LABEL_H + BORDER), dsts, prog)
            writer.append_data(np.asarray(img))

        # phase C: slide the next generation up into the playing slot
        n_c = int(0.5 * FPS)
        for f in range(1, n_c + 1):
            s = f / n_c
            s = s * s * (3 - 2 * s)
            dy = int(round(s * (bot_y - top_y)))
            img = Image.new("RGB", (W, H), BG)
            draw_grid(img, grid, 0, bot_y - dy, mid, nxt["fitness"], n_borders, n_labels, font)
            d = ImageDraw.Draw(img)
            header(d, g + 1, int(np.argmax(nxt["fitness"])))
            writer.append_data(np.asarray(img))
        prev_best = i_best

    writer.close()
    print("wrote", out)


def paste_tile(img, box, tile, border, label, font):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    img.paste(Image.fromarray(tile), (x0, y0))
    d.rectangle([x0, y1, x1, y1 + LABEL_H], fill=PANEL)
    d.text((x0 + 4, y1 + 2), label, fill=TEXT, font=font)
    col, wd = border
    d.rectangle([x0 - wd, y0 - wd, x1 + wd, y1 + LABEL_H + wd], outline=col, width=wd)


if __name__ == "__main__":
    main()
