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

TERRAINS = ["flat", "cobblestone", "potholes", "gravel", "slope"]

SLOPE_X0 = 0.15     # the slope begins this far ahead of the start; flat behind it
SLOPE_BLEND = 0.4   # and reaches its full grade over this length (a smooth knee, no edge)


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


def slope_height(x, y, slope_deg: float, heading_deg: float = 0.0):
    """Ground elevation of the "slope" terrain: flat for x < SLOPE_X0, then a plane of grade slope_deg whose uphill
    direction is heading_deg from +x (0 uphill ahead, 180 downhill ahead, 90 rising to the robot's left, -90 to its
    right), blended in over SLOPE_BLEND so the start has no edge. 0 at the origin, where the robot starts."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    u = np.clip((x - SLOPE_X0) / SLOPE_BLEND, 0.0, 1.0)
    w = u * u * (3 - 2 * u)                                    # smoothstep 0 -> 1
    g, h = np.tan(np.radians(slope_deg)), np.radians(heading_deg)
    along = np.where(x > SLOPE_X0, x - SLOPE_X0, 0.0)
    # the along-x part integrates the blend so the grade (not the height) ramps in
    ramp = np.where(u < 1, SLOPE_BLEND * (u ** 3 - u ** 4 / 2), SLOPE_BLEND / 2 + (along - SLOPE_BLEND))
    return g * (np.cos(h) * ramp + np.sin(h) * y * w)


def build(kind: str, rng: np.random.Generator, slope_deg: float = 0.0, slope_heading_deg: float = 0.0):
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

    if kind == "slope":
        # a heightfield plane (exact on the grid's triangles) placed so the start is at z = 0; the robot spawns as on
        # flat ground (TERRAIN_ZMAX 0) and the rollout reads the ground height from slope_height()
        y, x = np.mgrid[0:HF_ROWS, 0:HF_COLS]
        xs = -HF_HALF_X + 2 * HF_HALF_X * x / (HF_COLS - 1)
        ys = -HF_HALF_Y + 2 * HF_HALF_Y * y / (HF_ROWS - 1)
        h = slope_height(xs, ys, slope_deg, slope_heading_deg)
        lo, hi = float(h.min()), float(h.max())
        span = max(hi - lo, 1e-4)
        asset = (GRID_ASSET + f'<hfield name="slope" nrow="{HF_ROWS}" ncol="{HF_COLS}" '
                 f'size="{HF_HALF_X} {HF_HALF_Y} {span:.5f} 0.05"/>')
        geom = (f'<geom name="floor" type="hfield" hfield="slope" pos="0 0 {lo:.5f}" friction="1.0 0.005 0.0001" '
                f'condim="3" contype="3" conaffinity="1" material="grid" rgba="0.75 0.7 0.65 1"/>')
        return asset, geom, ((h - lo) / span).astype(np.float32)

    raise ValueError(f"unknown terrain {kind!r}; choose from {TERRAINS}")
