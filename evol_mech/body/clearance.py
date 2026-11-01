"""Interference check: leg links against the core and the fans.

Both linkage pivots sit on the hip (B on a post rising from it), so the rocker
BC and the rest of the leg sweep through space beside the core. The core yaws
+-30 deg and each fan pitches about its own axis, so a leg point interferes if
it lies inside the yaw-swept core box or the yaw-and-pitch-swept fan sphere.
Interfering bodies are rejected like non-rotatable linkages.
"""
from __future__ import annotations

import math

from . import linkage
from .structure import HIP_HALF_Z, LINK_RADIUS, core_half_x, core_half_y, fan_center_y

FAN_THICK = 0.008
MARGIN = 0.003
YAW_RANGE = math.radians(30)


def fan_diam_m(p: dict[str, float]) -> float:
    return p["fan_diam_in"] * 0.0254


def _segments(p: dict[str, float], theta: float):
    s = linkage.solve(p, theta)
    if s is None:
        return []
    A, B = (0.0, 0.0), (p["pivot_b_dx"], p["pivot_b_dz"])
    return [(A, s.D), (s.D, s.C), (s.C, s.P), (s.P, s.D), (B, s.C), (A, B)]


def interference(p: dict[str, float], yaw_range: float = 0.0, n_theta: int = 36, n_yaw: int = 7, n_pts: int = 6):
    """Return (reason, point) for the first interfering leg point, or None.
    yaw_range: half-range the core can yaw (0 when the waist is fixed)."""
    if yaw_range <= 0:
        n_yaw = 1
    y = p["hip_width"] / 2
    core_z0, core_z1 = HIP_HALF_Z - LINK_RADIUS - MARGIN, HIP_HALF_Z + p["core_height"] + LINK_RADIUS + MARGIN
    core_hx = core_half_x(p) + LINK_RADIUS + MARGIN
    core_hy = core_half_y(p) + LINK_RADIUS + MARGIN
    fan_r = fan_diam_m(p) / 2 + FAN_THICK + LINK_RADIUS + MARGIN
    fy = fan_center_y(p)
    fans = [(p["fan_x"], sgn * fy, HIP_HALF_Z + p["fan_z"]) for sgn in (1.0, -1.0)]  # fan_z from core bottom
    yaws = [0.0] if n_yaw == 1 else [-yaw_range + 2 * yaw_range * k / (n_yaw - 1) for k in range(n_yaw)]
    for k in range(n_theta):
        for a, b in _segments(p, 2 * math.pi * k / n_theta):
            for j in range(n_pts + 1):
                t = j / n_pts
                x, z = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
                for phi in yaws:                      # leg point expressed in the yawed core frame
                    c, s_ = math.cos(phi), math.sin(phi)
                    xr, yr = x * c + y * s_, -x * s_ + y * c
                    if abs(xr) < core_hx and abs(yr) < core_hy and core_z0 < z < core_z1:
                        return "leg hits core", (x, z)
                    for fx, fy, fz in fans:
                        if (xr - fx) ** 2 + (yr - fy) ** 2 + (z - fz) ** 2 < fan_r ** 2:
                            return "leg hits fan", (x, z)
    return None


def is_buildable(p: dict[str, float], waist_yaw: bool = False) -> bool:
    return linkage.is_valid(p) and interference(p, YAW_RANGE if waist_yaw else 0.0) is None
