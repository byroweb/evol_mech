"""Planar crank-rocker 4-bar geometry for one leg.

Labels follow the design sketch. Frame: leg sagittal plane, x forward, z up,
origin at the crank pivot.
    A = (0, 0)                    crank pivot, fixed to hip (stepper output)
    B = (pivot_b_dx, pivot_b_dz)  rocker pivot, fixed to hip
    D = crank tip                 |AD| = crank_len
    C = coupler/rocker joint      |CD| = coupler_len, |BC| = rocker_len
    P = foot point, rigid to the coupler: a triangle C-D-P with P placed
        foot_frac of the way from D to C and foot_height perpendicular to CD,
        on the side away from pivot A (i.e. below the coupler when standing).

Crank angle theta follows MuJoCo's hinge-about-+y convention, so that the
crank joint at qpos=0 reproduces the pose solved here for theta0:
    D(theta) = (-r sin theta, -r cos theta)        theta=0 -> straight down
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LegPose:
    D: tuple[float, float]      # crank tip
    C: tuple[float, float]      # coupler/rocker joint
    P: tuple[float, float]      # foot


def crank_tip(r: float, theta: float) -> tuple[float, float]:
    return (-r * math.sin(theta), -r * math.cos(theta))


def _circle_intersect(c1, r1, c2, r2):
    """Both intersection points of two circles, or None."""
    dx, dz = c2[0] - c1[0], c2[1] - c1[1]
    d = math.hypot(dx, dz)
    if d < 1e-9 or d > r1 + r2 or d < abs(r1 - r2):
        return None
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h2 = r1 * r1 - a * a
    if h2 < 0:
        return None
    h = math.sqrt(h2)
    mx, mz = c1[0] + a * dx / d, c1[1] + a * dz / d
    return ((mx + h * dz / d, mz - h * dx / d), (mx - h * dz / d, mz + h * dx / d))


def _orient(B, C, D) -> float:
    """Sign of the B-C-D triangle orientation; constant along one assembly branch."""
    return math.copysign(1.0, (C[0] - B[0]) * (D[1] - B[1]) - (C[1] - B[1]) * (D[0] - B[0]))


def _reference(p: dict[str, float]):
    """Assembly branch and foot side, fixed once at theta=0 so the foot is a
    rigid part of the coupler and the rocker never jumps branches."""
    r, L, q = p["crank_len"], p["coupler_len"], p["rocker_len"]
    B = (p["pivot_b_dx"], p["pivot_b_dz"])
    D = crank_tip(r, 0.0)
    pts = _circle_intersect(D, L, B, q)
    if pts is None:
        return None
    C = min(pts, key=lambda c: c[1])              # lower assembly at theta=0
    ux, uz = (C[0] - D[0]) / L, (C[1] - D[1]) / L
    nx, nz = uz, -ux                              # fixed handedness
    side = -math.copysign(1.0, nx * (0 - D[0]) + nz * (0 - D[1]))   # away from A
    return _orient(B, C, D), side


def solve(p: dict[str, float], theta: float) -> LegPose | None:
    """Closed-loop pose at crank angle theta, or None if the loop cannot close."""
    ref = _reference(p)
    if ref is None:
        return None
    branch, side = ref
    r, L, q = p["crank_len"], p["coupler_len"], p["rocker_len"]
    B = (p["pivot_b_dx"], p["pivot_b_dz"])
    D = crank_tip(r, theta)
    pts = _circle_intersect(D, L, B, q)
    if pts is None:
        return None
    same = [c for c in pts if _orient(B, c, D) == branch]
    C = same[0] if same else min(pts, key=lambda c: c[1])
    ux, uz = (C[0] - D[0]) / L, (C[1] - D[1]) / L
    nx, nz = uz, -ux
    f, h = p["foot_frac"] * L, side * p["foot_height"]
    P = (D[0] + f * ux + h * nx, D[1] + f * uz + h * nz)
    return LegPose(D, C, P)


def is_valid(p: dict[str, float], samples: int = 72) -> bool:
    """Grashof crank-rocker with the crank as shortest link, and closable at
    every sampled crank angle. Cheap enough to run before every rollout."""
    r, L, q = p["crank_len"], p["coupler_len"], p["rocker_len"]
    g = math.hypot(p["pivot_b_dx"], p["pivot_b_dz"])
    links = sorted([r, L, q, g])
    if links[0] != r:
        return False
    if links[0] + links[3] > links[1] + links[2] - 1e-6:
        return False
    for k in range(samples):
        if solve(p, 2 * math.pi * k / samples) is None:
            return False
    return True


def foot_path(p: dict[str, float], samples: int = 72) -> np.ndarray:
    """(samples, 2) array of foot positions over one crank revolution."""
    out = np.zeros((samples, 2))
    for k in range(samples):
        pose = solve(p, 2 * math.pi * k / samples)
        out[k] = np.nan if pose is None else pose.P
    return out


def lowest_foot_z(p: dict[str, float]) -> float:
    return float(np.nanmin(foot_path(p)[:, 1]))


def parking_angle(p: dict[str, float], phase: float, samples: int = 144) -> float:
    """Left crank angle at which the two feet (right crank = left + phase) are
    closest to level and centred under pivot A, preferring low stances. Used
    to park the steppers for the standing task."""
    best, best_cost = 0.0, math.inf
    for k in range(samples):
        th = 2 * math.pi * k / samples
        a, b = solve(p, th), solve(p, th + phase)
        if a is None or b is None:
            continue
        cost = abs(a.P[1] - b.P[1]) + 0.6 * abs(a.P[0] + b.P[0]) / 2 - 0.1 * min(-a.P[1], -b.P[1])
        if cost < best_cost:
            best, best_cost = th, cost
    return best
