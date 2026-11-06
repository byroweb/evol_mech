"""Terrain generators. Each returns (asset_xml, geom_xml, hfield_data_or_None).

Heightfields are declared in the XML with a fixed grid and filled in Python
after compilation (model.hfield_data), which keeps the XML small and makes the
terrain seedable.
"""
from __future__ import annotations

import numpy as np

HF_ROWS = 200
HF_COLS = 200
HF_HALF_X = 6.0   # metres, field spans [-6, 6]
HF_HALF_Y = 3.0

TERRAINS = ["flat", "cobblestone", "potholes", "gravel"]


# 10 cm checker so motion is readable in the viewer; texuniform makes texrepeat per metre
GRID_ASSET = ('<texture name="grid" type="2d" builtin="checker" width="256" height="256" '
              'rgb1="0.82 0.82 0.82" rgb2="0.55 0.55 0.58"/>'
              '<material name="grid" texture="grid" texrepeat="10 10" texuniform="true" reflectance="0.1"/>')


def _hfield_xml(name: str, zmax: float, friction: str) -> tuple[str, str]:
    asset = (GRID_ASSET + f'<hfield name="{name}" nrow="{HF_ROWS}" ncol="{HF_COLS}" '
             f'size="{HF_HALF_X} {HF_HALF_Y} {zmax} 0.05"/>')
    geom = (f'<geom name="floor" type="hfield" hfield="{name}" pos="0 0 0" '
            f'friction="{friction}" condim="3" contype="3" conaffinity="1" material="grid" rgba="0.75 0.7 0.65 1"/>')
    return asset, geom


def height_at(hdata, zmax: float, x: float, y: float) -> float:
    """Ground elevation at (x, y) for a heightfield from build() (0 for flat)."""
    if hdata is None:
        return 0.0
    nrow, ncol = hdata.shape
    c = int(round((x + HF_HALF_X) / (2 * HF_HALF_X) * (ncol - 1)))
    r = int(round((y + HF_HALF_Y) / (2 * HF_HALF_Y) * (nrow - 1)))
    return float(hdata[min(max(r, 0), nrow - 1), min(max(c, 0), ncol - 1)]) * zmax


def build(kind: str, rng: np.random.Generator):
    if kind == "flat":
        geom = ('<geom name="floor" type="plane" size="0 0 0.05" '
                'friction="1.0 0.005 0.0001" contype="3" conaffinity="1" material="grid"/>')
        return GRID_ASSET, geom, None

    if kind == "cobblestone":
        # 4-7 cm stones, 1-2 cm relief: rounded bumps on a grid with jitter
        zmax = 0.02
        asset, geom = _hfield_xml("cobble", zmax, "1.0 0.005 0.0001")
        y, x = np.mgrid[0:HF_ROWS, 0:HF_COLS]
        cell = 2.0  # grid cells per stone (~6 cm at 200 cells / 12 m)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        h = 0.5 * (1 + np.sin(2 * np.pi * x / cell + phase_x)) * \
            0.5 * (1 + np.sin(2 * np.pi * y / cell + phase_y))
        h += rng.uniform(0, 0.25, h.shape)
        h = np.clip(h / h.max(), 0, 1)
        return asset, geom, h.astype(np.float32)

    if kind == "potholes":
        # flat slab with randomly placed pits 10-30 cm wide, 3 cm deep
        zmax = 0.03
        asset, geom = _hfield_xml("potholes", zmax, "1.0 0.005 0.0001")
        h = np.ones((HF_ROWS, HF_COLS), dtype=np.float32)
        y, x = np.mgrid[0:HF_ROWS, 0:HF_COLS]
        for _ in range(40):
            cx, cy = rng.uniform(0, HF_COLS), rng.uniform(0, HF_ROWS)
            r = rng.uniform(2, 5)  # cells
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            h[d2 < r * r] = 0.0
        return asset, geom, h

    if kind == "gravel":
        # approximation: fine high-frequency noise + lower friction.
        # true granular media is deliberately out of scope for the GA loop.
        zmax = 0.012
        asset, geom = _hfield_xml("gravel", zmax, "0.6 0.005 0.0001")
        h = rng.uniform(0, 1, (HF_ROWS, HF_COLS)).astype(np.float32)
        # light smoothing so it is lumpy, not spiky
        h = (h + np.roll(h, 1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 0) + np.roll(h, -1, 1)) / 5
        return asset, geom, h

    raise ValueError(f"unknown terrain {kind!r}; choose from {TERRAINS}")
