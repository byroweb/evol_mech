"""Static feasibility of a skater body, no simulation (docs/skater_test_plan.md, phase 0).

A body that passes can at least stand on one leg anywhere a walking step needs it:

  workspace   how far the ankle can lift, reach fore and aft and extend from the parked
              crouch before the leg's own linkage runs out (hard stops, folds, reach)
  torque      the leg servos' torque to hold the whole body on one leg at the worst of
              those poses, as a fraction of stall, from the leg's Jacobian; the roll servo's
              torque to hold the body over a foot placed at the standing track, through
              the bracket lever
  roll range  the hip roll the standing track needs against the roll servo's range
  soles       the two soles' gap at the standing track (the controller clamps inboard
              feet to this; the check reports the body that asked for less)

`static_check(p)` returns the numbers and the reasons it fails; `feasible(p)` the verdict. The
body rounds of phase 2 sample only bodies that pass. Thresholds are a walking step's
(the 4 cm lift and 6 cm fore / aft that `check_buildable` already demands), a static
torque under STATIC_TORQUE_MAX of stall (the plan's dynamic bound at touchdown is 60 %),
and nothing on a hard stop.

    python -m evol_mech.skater.feasibility configs/archetypes/skater_walker_panto9.json
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from ..body.hardware import roll_servo_from_params, servo_from_params
from .structure import (InvalidBody, angles_from_foot, check_buildable, drive_pose_allowed, foot_from_angles,
                        hip_roll_range, horn_from_shin, is_pantograph, nominal_angles, nominal_foot, panto_ik,
                        panto_nominal, panto_points, panto_pose_allowed, roll_arm, roll_ratio, shin_from_horn,
                        sole_half_w, structure_from_params)

G = 9.81
STEP = 0.005                 # workspace probing step (m)
LIFT_MIN, FORE_MIN, AFT_MIN = 0.04, 0.06, 0.06     # a walking step's workspace, as check_buildable asks
STATIC_TORQUE_MAX = 0.8      # of stall, holding the body on one leg at the worst pose
SHEAR_FRAC = 0.3             # fore-aft ground force on top of the weight (acceleration, braking)
SOLE_GAP_MIN = 0.01          # between the two soles at the standing track (the controller's clamp)


def _solve(p: dict[str, float], x: float, z: float, guess) -> tuple[float, float] | None:
    """Servo angles (horn_f, horn_r) or (thigh, horn) that put the ankle at (x, z) in the leg
    plane, None when the leg cannot: out of reach, or a link within its fold / stop margins."""
    if is_pantograph(p):
        a, b, res = panto_ik(p, x, z, guess)
        return (a, b) if res < 1e-3 and panto_pose_allowed(p, a, b) else None
    thigh, shin = angles_from_foot(p, x, z)
    fx, fz = foot_from_angles(p, thigh, shin)
    if math.hypot(fx - x, fz - z) > 1e-3:              # reach clamped
        return None
    horn = horn_from_shin(p, thigh, shin)
    if horn is None or not drive_pose_allowed(p, thigh, horn):
        return None
    return thigh, horn


def _ankle(p: dict[str, float], a: float, b: float) -> tuple[float, float]:
    if is_pantograph(p):
        k = panto_points(p, a, b)
        return k["ankle"] if k is not None else (math.nan, math.nan)
    r = shin_from_horn(p, a, b)
    if r is None:
        return math.nan, math.nan
    return foot_from_angles(p, a, r[0])


def _extent(p: dict[str, float], x0: float, z0: float, dx: float, dz: float, guess) -> float:
    """How far the ankle can go from (x0, z0) along (dx, dz) in STEP increments, metres."""
    d, g = 0.0, guess
    while d < 0.5:
        sol = _solve(p, x0 + (d + STEP) * dx, z0 + (d + STEP) * dz, g)
        if sol is None:
            break
        d, g = d + STEP, sol
    return d


def _torque_frac(p: dict[str, float], pose, weight: float, stall_nm: float) -> float:
    """Leg servo torque, as a fraction of stall, to hold the body's whole weight on this leg at
    `pose` with a fore-aft shear of SHEAR_FRAC x weight either way: tau = J^T F over the leg's
    2 x 2 Jacobian (finite differences of the ankle on the servo angles)."""
    a, b = pose
    h = 1e-4
    x0 = np.array(_ankle(p, a, b))
    J = np.column_stack([(np.array(_ankle(p, a + h, b)) - x0) / h, (np.array(_ankle(p, a, b + h)) - x0) / h])
    if not np.all(np.isfinite(J)):
        return math.inf
    worst = 0.0
    for shear in (-SHEAR_FRAC, SHEAR_FRAC):
        F = weight * np.array([shear, 1.0])            # the ground's force on the foot, (x, z)
        worst = max(worst, float(np.max(np.abs(J.T @ F))) / stall_nm)
    return worst


def _nominal_pose(p: dict[str, float]):
    if is_pantograph(p):
        return panto_nominal(p)
    thigh, shin = nominal_angles(p)
    horn = horn_from_shin(p, thigh, shin)
    return None if horn is None else (thigh, horn)


def static_check(p: dict[str, float]) -> dict:
    out: dict = {"ok": True, "reasons": []}

    def fail(reason: str) -> None:
        out["ok"] = False
        out["reasons"].append(reason)

    try:
        check_buildable(p)
    except InvalidBody as e:
        fail(f"not buildable: {e}")
        return out
    st = structure_from_params(p)
    weight = st.total_kg * G
    servo, rservo = servo_from_params(p), roll_servo_from_params(p)
    x_nom, z_nom = nominal_foot(p)
    nom = _nominal_pose(p)
    if nom is None or _solve(p, x_nom, z_nom, nom) is None:
        fail("the parked crouch is not a reachable pose")
        return out

    # workspace from the crouch
    lift = _extent(p, x_nom, z_nom, 0.0, 1.0, nom)
    fore = _extent(p, x_nom, z_nom, 1.0, 0.0, nom)
    aft = _extent(p, x_nom, z_nom, -1.0, 0.0, nom)
    ext = _extent(p, x_nom, z_nom, 0.0, -1.0, nom)
    out.update(lift_m=lift, fore_m=fore, aft_m=aft, extend_m=ext, crouch_drop_m=-z_nom)
    if lift < LIFT_MIN:
        fail(f"lift {lift * 100:.1f} cm < {LIFT_MIN * 100:.0f}")
    if fore < FORE_MIN:
        fail(f"fore reach {fore * 100:.1f} cm < {FORE_MIN * 100:.0f}")
    if aft < AFT_MIN:
        fail(f"aft reach {aft * 100:.1f} cm < {AFT_MIN * 100:.0f}")

    # leg servo torque at the worst standing pose: the crouch, the fore and aft reach, half the lift
    fo, af, li = min(fore, FORE_MIN), min(aft, AFT_MIN), min(lift, LIFT_MIN)
    poses = [(x_nom, z_nom), (x_nom + fo, z_nom), (x_nom - af, z_nom), (x_nom, z_nom + li / 2), (x_nom + fo / 2, z_nom + li / 2)]
    worst, worst_pose = 0.0, None
    for x, z in poses:
        sol = _solve(p, x, z, nom)
        if sol is None:
            continue
        f = _torque_frac(p, sol, weight, servo.torque_nm)
        if f > worst:
            worst, worst_pose = f, (round(x - x_nom, 3), round(z - z_nom, 3))
    out.update(leg_torque_frac=worst, leg_torque_pose=worst_pose)
    if worst > STATIC_TORQUE_MAX:
        fail(f"leg servo at {worst * 100:.0f} % of stall holding one-legged at {worst_pose}")

    # the standing track: the foot stance_w outboard (+) / inboard (-) of the leg's hip point
    gap = p["hip_width"] + 2 * p["stance_w"] - 2 * sole_half_w(p)
    stance_w = max(p["stance_w"], -(p["hip_width"] - p["tread_w"] - SOLE_GAP_MIN) / 2)
    out.update(sole_gap_m=gap, stance_w_used=stance_w)
    if gap < SOLE_GAP_MIN:
        fail(f"soles {gap * 100:.1f} cm apart at the asked track (the controller clamps stance_w to {stance_w:.3f})")
    # the roll that puts the ankle stance_w to the side and the crouch's drop below the hip point
    # (the controller's formula, about the pivot roll_arm inboard of that point), against the range
    h, L = -z_nom, roll_arm(p)
    if L == 0.0:
        yt = stance_w
        roll_need = math.atan2(yt, h)
    else:
        yt = L + stance_w
        drop = math.sqrt(max(1e-4, yt * yt + h * h - L * L))
        roll_need = math.atan2(yt, h) - math.atan2(L, drop)
    rng = hip_roll_range(p)
    out.update(roll_need_deg=math.degrees(abs(roll_need)), roll_range_deg=math.degrees(rng),
               roll_margin_deg=math.degrees(rng - abs(roll_need)))
    if abs(roll_need) > rng:
        fail(f"standing track needs {math.degrees(abs(roll_need)):.1f} deg of hip roll, range {math.degrees(rng):.1f}")
    # the roll servo holding the body over that foot: weight x the foot's lateral arm about the
    # pivot, through the bracket lever onto the servo
    ratio = roll_ratio(p)
    roll_frac = weight * abs(yt) / (rservo.torque_nm * ratio) if ratio > 0 else math.inf
    out.update(roll_torque_frac=roll_frac)
    if roll_frac > STATIC_TORQUE_MAX:
        fail(f"roll servo at {roll_frac * 100:.0f} % of stall holding the body over the standing foot")
    out.update(total_kg=st.total_kg, servo_kgcm=servo.stall_kgcm, roll_servo_kgcm=rservo.stall_kgcm)
    return out


def feasible(p: dict[str, float]) -> bool:
    return bool(static_check(p)["ok"])


def main(argv: list[str]) -> None:
    for path in argv:
        from .spec import decode, encode
        d = json.load(open(path, encoding="utf-8"))
        p = decode(encode(d.get("phys", d)))             # older archetypes lack the newer genes
        r = static_check(p)
        print(path)
        for k, v in r.items():
            if k != "reasons":
                print(f"  {k:18s} {v:.4f}" if isinstance(v, float) else f"  {k:18s} {v}")
        for why in r["reasons"]:
            print("  FAIL:", why)


if __name__ == "__main__":
    main(sys.argv[1:])
