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

HIP_HALF_Z = 0.035          # hip box below/above crank pivot A (compact; B is on a post)
CORE_HALF_XY = 0.05
FAN_THICK = 0.008
LINK_RADIUS = 0.006
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


def interference(p: dict[str, float], n_theta: int = 36, n_yaw: int = 7, n_pts: int = 6):
    """Return (reason, point) for the first interfering leg point, or None."""
    y = p["hip_width"] / 2
    core_z = HIP_HALF_Z + p["core_height"] / 2
    core_z0, core_z1 = HIP_HALF_Z - LINK_RADIUS - MARGIN, HIP_HALF_Z + p["core_height"] + LINK_RADIUS + MARGIN
    core_half = CORE_HALF_XY + LINK_RADIUS + MARGIN
    fan_r = fan_diam_m(p) / 2 + FAN_THICK + LINK_RADIUS + MARGIN
    fans = [(p["fan_x"], sgn * p["fan_y"], core_z + p["fan_z"]) for sgn in (1.0, -1.0)]
    yaws = [-YAW_RANGE + 2 * YAW_RANGE * k / (n_yaw - 1) for k in range(n_yaw)]
    for k in range(n_theta):
        for a, b in _segments(p, 2 * math.pi * k / n_theta):
            for j in range(n_pts + 1):
                t = j / n_pts
                x, z = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
                for phi in yaws:                      # leg point expressed in the yawed core frame
                    c, s_ = math.cos(phi), math.sin(phi)
                    xr, yr = x * c + y * s_, -x * s_ + y * c
                    if abs(xr) < core_half and abs(yr) < core_half and core_z0 < z < core_z1:
                        return "leg hits core", (x, z)
                    for fx, fy, fz in fans:
                        if (xr - fx) ** 2 + (yr - fy) ** 2 + (z - fz) ** 2 < fan_r ** 2:
                            return "leg hits fan", (x, z)
    return None


def is_buildable(p: dict[str, float]) -> bool:
    return linkage.is_valid(p) and interference(p) is None
