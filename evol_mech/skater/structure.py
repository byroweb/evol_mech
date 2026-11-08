"""Skater mech (archetype B): derived masses, leg kinematics and foot layout.
Never genes.

Mass rules from docs/skater_design.md:
  - every leg servo sits in the hip (three per leg: roll, thigh, shin);
  - below the hip are tubes and the parallelogram links, the tread drive
    (motor and worm) on the shin at the knee, the sole, rollers and rubber;
  - each leg's distal mass should stay at or below ~10-12 % of total mass;
    `leg_mass_fraction` reports it and the rollout penalises the excess.

Leg kinematics (sagittal plane, angles relative to the body, positive = link
swung backward): the thigh servo sets the thigh angle, the shin servo sets
the shin angle through a four-bar: horn on the shin servo -> coupler -> pin
tab on the shin tube. With the shin servo coaxial with the hip pitch axis
and equal horn / crank it is LEO's parallelogram (shin angle = horn angle);
the default staggers the shin servo aft (tools/mechanism_checker.html), so
the shin angle depends a little on the thigh angle too, and the drive has
hard stops where a link would come within FOLD_MARGIN of straight. The knee
hinge angle is shin minus thigh. The ankle is a passive two-axis sprung hinge
(pitch and roll), so the sole is not part of the servo kinematics: it lies
flat under load and rests toe-up when unloaded. The hip roll servo (LEO's
abduction axis, along x through the hip point) swings the whole leg plane
sideways; the controller puts the stance foot under the body with it.

Pantograph leg (leg_drive 1, the user's 2026-11-07 sketch): both leg servos
sit under the hip bracket at the same height, `servo_dx` apart. Each horn
(`horn_len`) drives a rod (`rod_len`) to a common pin, the knee node, where
the two rods cross and are pinned together; each rod continues `rod_ext`
past the pin (the front servo's rod to the rear lower node, the rear
servo's to the front lower node) and two links of the same length close a
parallelogram at the ankle node, so the ankle node sits at the vector sum of
the two rod extensions. A strut of `strut_len` continues the rear rod's
direction from the ankle node to the ankle, the same passive sprung foot as
the four-bar leg. The two horn angles set where the pin sits (a five-bar)
and the pantograph carries that to the foot: both servos share every load.
The hip roll is a lever. On the pantograph body the two roll pivots sit at
the hip centre (ROLL_PIVOT_Y apart, the sketch's two bolts) and each bracket
runs out to its leg plane, so the bracket is the lever arm (hip_width / 2 -
ROLL_PIVOT_Y, `roll_arm`) and the roll ratio is that over the horn length
(`roll_ratio`, derived); the roll servos sit at the centre, horn horizontal
and a vertical link down to the bracket's outboard end. Rolling the leg also
raises or lowers the leg point by arm x sin(roll), which the controller's
roll solve accounts for. On the four-bar body the pivot is at the leg plane
and the `roll_ratio` gene stands in for an arm. All three horns are one part
(`horn_len`) and flex in their sweep plane (`horn_k_log10`), a torsion spring
in series with the servo, stiff out of plane so the kinematics stay planar.

The core, battery and fan placement rules are shared with the biped
(`body/structure.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..body.hardware import (fan_from_params, servo_from_params, roll_servo_from_params, tread_drive_from_params,
                             TREAD_RUBBER_KG_PER_M2, ROLLER_KG_PER_M, SOLE_PLATE_KG_PER_M2)
from ..body.structure import (HIP_HALF_X, HIP_HALF_Z, SHELL_KG_PER_M2, HIP_HARDWARE_KG,
                              FAN_ARM_KG_PER_M, ELECTRONICS_KG, BATTERY_KG_BASE, BATTERY_KG_PER_MAH,
                              hip_half_y, core_half_x, core_half_y, payload_z, fan_center_y,
                              fan_z_core, battery_half, _box_area)

LINK_RADIUS = 0.008             # thigh / shin tube radius (16 mm CF tube)
LINK_KG_PER_M = 0.08            # tube + bearings, per metre
COUPLER_KG_PER_M = 0.05         # parallelogram coupler (thin CF strip + pins), per metre of thigh
JOINT_BRACKET_KG = 0.025        # knee and ankle brackets, each
ANKLE_SPRING_KG = 0.010         # per axis (pitch and roll)
SOLE_HALF_W = 0.020             # sole plate half-width
SOLE_HALF_T = 0.004             # sole plate half-thickness
ANKLE_PLATE_GAP = 0.006         # ankle axis above the plate top (the coaxial pulley sits there)
ROLLER_HALF_LEN = 0.015         # tread width 30 mm (the default of the tread_w gene)


def roller_half_len(p: dict[str, float]) -> float:
    """Half the tread width: the rollers' half-length, the foot's support half-width in roll."""
    return float(p.get("tread_w", 2 * ROLLER_HALF_LEN)) / 2


def sole_half_w(p: dict[str, float]) -> float:
    """The sole plate's half-width: at least the tread's."""
    return max(SOLE_HALF_W, roller_half_len(p))
ROLLER_GAP = 0.002              # clearance between adjacent rollers
MAX_ROLLERS = 6
IDLER_RISE = 0.6                # front idler axis raised by this fraction of a roller radius
WHEEL_RADIUS_FACTOR = 1.5       # foot_type 1: heel and toe wheels, larger than the tread rollers
LEG_MASS_FRACTION_TARGET = 0.12
SERVOS_PER_LEG = 3

HIP_ROLL_RANGE = math.radians(30)    # abduction: + swings the foot to +y (left) on both legs
ANKLE_ROLL_RANGE = math.radians(30)  # passive sole roll, so a rolled leg still lands the tread flat
ANKLE_ROLL_STIFF = 0.5               # ankle roll spring as a fraction of the pitch spring
THIGH_RANGE = (math.radians(-120), math.radians(60))    # negative = thigh forward
KNEE_RANGE = (math.radians(10), math.radians(150))      # positive = shin back; the shin drive never comes within 10 deg of straight
ANKLE_RANGE = math.radians(45)

# shin drive: the staggered four-bar of tools/mechanism_checker.html (its defaults)
SHIN_DRIVE = {
    "dx": -0.045, "dz": 0.0,          # shin servo axis from the hip pitch axis (x forward, z up)
    "horn": 0.040, "crank": 0.041,    # servo horn, and knee-to-pin distance along the shin tab
    "crank_angle": math.radians(15),  # pin tab aft of the shin axis
    "coupler": None,                  # None = auto: horn parallel to the shin at the nominal crouch
}
FOLD_MARGIN = math.radians(10)        # no two links of the shin drive within this of straight
HORN_RANGE = (math.radians(-130), math.radians(130))
HORN_KG = 0.010                       # pantograph: a servo horn with its rod-end bearing
ROLL_PIVOT_Y = 0.012                  # pantograph body: each roll pivot this far from the hip centre (two bolts side by side)
PIN_KG = 0.008                        # pantograph: a pinned node (bolt, bushing, spacer)


class InvalidBody(ValueError):
    """Raised when the genome decodes to a body that cannot be built."""


# ---- foot layout ----------------------------------------------------------------

@dataclass(frozen=True)
class FootLayout:
    roller_r: float
    roller_x: tuple[float, ...]     # roller centres along the sole, ankle at x = 0
    idler: tuple[float, float] | None   # raised front idler (x, dz above the roller axis), tread only
    plate_z: float                  # sole plate centre below the ankle axis
    roller_z: float                 # roller axis below the ankle axis
    bottom_z: float                 # lowest point below the ankle axis


def foot_layout(p: dict[str, float]) -> FootLayout:
    L, r = p["sole_len"], p["pulley_radius"]
    idler = None
    if p["foot_type"] >= 0.5:                      # heel + toe wheels
        rw = min(WHEEL_RADIUS_FACTOR * r, (L - ROLLER_GAP) / 4)
        xs = (-(L / 2 - rw), L / 2 - rw)
        r = rw
    else:                                          # row of tread rollers + raised front idler
        if L < 4 * r + ROLLER_GAP:
            raise InvalidBody(f"sole {L*1e3:.0f} mm too short for {r*1e3:.0f} mm rollers")
        n = min(MAX_ROLLERS, int((L - 2 * r) / (2 * r + ROLLER_GAP)) + 1)
        xs = tuple(-(L / 2 - r) + (L - 2 * r) * k / (n - 1) for k in range(n))
        idler = (L / 2 + r + ROLLER_GAP, IDLER_RISE * r)
    plate_z = -(ANKLE_PLATE_GAP + SOLE_HALF_T)
    roller_z = plate_z - SOLE_HALF_T - r
    return FootLayout(r, xs, idler, plate_z, roller_z, roller_z - r)


# ---- leg kinematics (sagittal plane) -------------------------------------------

def nominal_angles(p: dict[str, float]) -> tuple[float, float]:
    """(thigh, shin) body-relative angles of the parked crouch: knee flexion =
    crouch, thigh chosen so the ankle sits directly below the hip."""
    L1, L2, k = p["thigh_len"], p["shin_len"], p["crouch"]
    thigh = -math.atan2(L2 * math.sin(k), L1 + L2 * math.cos(k))
    return thigh, thigh + k


def foot_from_angles(p: dict[str, float], thigh: float, shin: float) -> tuple[float, float]:
    """Ankle position (x forward, z up) relative to the hip joint."""
    L1, L2 = p["thigh_len"], p["shin_len"]
    return (-L1 * math.sin(thigh) - L2 * math.sin(shin),
            -L1 * math.cos(thigh) - L2 * math.cos(shin))


def angles_from_foot(p: dict[str, float], x: float, z: float) -> tuple[float, float]:
    """Two-link inverse kinematics: (thigh, shin) body-relative angles that put
    the ankle at (x, z) relative to the hip, knee forward. Reach is clamped."""
    L1, L2 = p["thigh_len"], p["shin_len"]
    d = math.hypot(x, z)
    d = min(max(d, abs(L1 - L2) + 1e-4), L1 + L2 - 1e-4)
    knee = math.acos(max(-1.0, min(1.0, (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2))))
    knee = math.pi - knee                                   # flexion: 0 = straight
    knee = min(max(knee, KNEE_RANGE[0]), KNEE_RANGE[1])
    # direction to the foot (angle from straight down, positive = backward)
    phi = math.atan2(-x, -z)
    # angle between the thigh and the hip->foot line
    alpha = math.atan2(L2 * math.sin(knee), L1 + L2 * math.cos(knee))
    thigh = phi - alpha
    return thigh, thigh + knee


def _u(th: float) -> tuple[float, float]:
    """Link direction (x, z) for a body-relative angle (0 = straight down, + = backward)."""
    return -math.sin(th), -math.cos(th)


def coupler_len(p: dict[str, float], mech: dict = SHIN_DRIVE) -> float:
    """Coupler length: the given value, or (auto) the length that makes the horn
    parallel to the shin at the nominal crouch. Coaxial servo + equal horn /
    crank + zero crank angle gives the thigh length: LEO's parallelogram."""
    if mech.get("coupler"):
        return float(mech["coupler"])
    tn, sn = nominal_angles(p)
    L1, h1, h2 = p["thigh_len"], mech["horn"], mech["crank"]
    uT, uS, uH = _u(tn), _u(sn + mech["crank_angle"]), _u(sn)
    qx, qz = L1 * uT[0] + h2 * uS[0], L1 * uT[1] + h2 * uS[1]
    px, pz = mech["dx"] + h1 * uH[0], mech["dz"] + h1 * uH[1]
    return math.hypot(qx - px, qz - pz)


def shin_from_horn(p: dict[str, float], thigh: float, horn: float,
                   mech: dict = SHIN_DRIVE) -> tuple[float, float, float] | None:
    """Solve the shin drive: (shin angle, horn-coupler angle, coupler-crank
    angle), or None when the coupler cannot reach. Angles body-relative. Of
    the two circle intersections the one nearest the parallelogram-like pose
    (horn parallel to the shin) is taken, as the checker does."""
    L1, h1, h2, Lc = p["thigh_len"], mech["horn"], mech["crank"], coupler_len(p, mech)
    uT, uH = _u(thigh), _u(horn)
    kx, kz = L1 * uT[0], L1 * uT[1]
    px, pz = mech["dx"] + h1 * uH[0], mech["dz"] + h1 * uH[1]
    dx, dz = px - kx, pz - kz
    d = math.hypot(dx, dz)
    if d < 1e-9 or d > h2 + Lc + 1e-9 or d < abs(h2 - Lc) - 1e-9:
        return None
    a = (h2 * h2 - Lc * Lc + d * d) / (2 * d)
    hh = math.sqrt(max(0.0, h2 * h2 - a * a))
    ex, ez = dx / d, dz / d
    mx, mz = kx + a * ex, kz + a * ez
    ug = _u(horn + mech["crank_angle"])
    gx, gz = kx + h2 * ug[0], kz + h2 * ug[1]
    best = None
    for sgn in (1.0, -1.0):
        qx, qz = mx - sgn * hh * ez, mz + sgn * hh * ex
        err = math.hypot(qx - gx, qz - gz)
        if best is None or err < best[0]:
            best = (err, qx, qz)
    _, qx, qz = best
    shin = math.atan2(-(qx - kx), -(qz - kz)) - mech["crank_angle"]
    shin = math.atan2(math.sin(shin), math.cos(shin))

    def ang(ax, az, bx, bz):
        c = (ax * bx + az * bz) / (math.hypot(ax, az) * math.hypot(bx, bz) + 1e-12)
        return math.acos(max(-1.0, min(1.0, c)))
    horn_coupler = ang(px - mech["dx"], pz - mech["dz"], qx - px, qz - pz)
    transmission = ang(qx - px, qz - pz, qx - kx, qz - kz)
    return shin, horn_coupler, transmission


def horn_from_shin(p: dict[str, float], thigh: float, shin: float,
                   mech: dict = SHIN_DRIVE) -> float | None:
    """Horn angle that puts the shin at `shin` for this thigh angle (Newton from
    the parallelogram guess horn = shin), or None if the drive cannot reach it."""
    phi, err = shin, float("inf")
    r = shin_from_horn(p, thigh, phi, mech)
    if r is None:                       # the guess itself is unreachable: look for a reachable start nearby
        for d in (0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8):
            r = shin_from_horn(p, thigh, phi + d, mech)
            if r is not None:
                phi += d
                break
        else:
            return None
    for _ in range(20):
        err = math.atan2(math.sin(r[0] - shin), math.cos(r[0] - shin))
        if abs(err) < 1e-6:
            return phi
        r2 = shin_from_horn(p, thigh, phi + 1e-4, mech)
        if r2 is None:
            r2 = shin_from_horn(p, thigh, phi - 1e-4, mech)
            if r2 is None:
                return None
            slope = (r[0] - r2[0]) / 1e-4
        else:
            slope = (r2[0] - r[0]) / 1e-4
        if abs(slope) < 1e-6:
            return None
        step = -max(-0.3, min(0.3, err / slope))
        for _ in range(6):                  # backtrack if the step leaves the reachable set
            rn = shin_from_horn(p, thigh, phi + step, mech)
            if rn is not None:
                break
            step *= 0.5
        else:
            return None
        phi, r = phi + step, rn
    return phi if abs(err) < 1e-3 else None


def drive_pose_allowed(p: dict[str, float], thigh: float, horn: float, mech: dict = SHIN_DRIVE) -> bool:
    """Hard-stop rule of the shin drive: reachable, knee flexion within
    KNEE_RANGE, thigh within THIGH_RANGE, and neither the horn-coupler nor the
    coupler-crank angle within FOLD_MARGIN of straight (0 or 180 deg). The
    checker adds geometric interference on top; this is the angular core."""
    if not THIGH_RANGE[0] - 1e-9 <= thigh <= THIGH_RANGE[1] + 1e-9:
        return False
    r = shin_from_horn(p, thigh, horn, mech)
    if r is None:
        return False
    shin, hc, mu = r
    knee = shin - thigh
    lo, hi = FOLD_MARGIN - 1e-9, math.pi - FOLD_MARGIN + 1e-9
    return (KNEE_RANGE[0] - 1e-9 <= knee <= KNEE_RANGE[1] + 1e-9) and lo <= hc <= hi and lo <= mu <= hi


@dataclass(frozen=True)
class DriveStops:
    """Horn hard stops per thigh angle: the widest contiguous allowed band of
    horn angles for each thigh on a 1 deg grid; NaN where the thigh has none."""
    thigh: tuple          # grid (rad)
    lo: tuple             # horn band low (rad) per thigh
    hi: tuple             # horn band high (rad) per thigh

    def band(self, thigh: float) -> tuple[float, float] | None:
        t = min(max(thigh, self.thigh[0]), self.thigh[-1])
        k = min(int(round((t - self.thigh[0]) / (self.thigh[1] - self.thigh[0]))), len(self.thigh) - 1)
        lo, hi = self.lo[k], self.hi[k]
        return None if math.isnan(lo) else (lo, hi)

    def thigh_range(self) -> tuple[float, float]:
        ok = [t for t, lo in zip(self.thigh, self.lo) if not math.isnan(lo)]
        return (ok[0], ok[-1]) if ok else (float("nan"), float("nan"))


def drive_stops(p: dict[str, float], mech: dict = SHIN_DRIVE, step_deg: float = 1.0,
                horn_step_deg: float = 0.5) -> DriveStops:
    """Scan the (thigh, horn) grid once per body and keep, for each thigh, the
    widest contiguous band of allowed horn angles."""
    thighs = [THIGH_RANGE[0] + math.radians(step_deg) * k
              for k in range(int(round(math.degrees(THIGH_RANGE[1] - THIGH_RANGE[0]) / step_deg)) + 1)]
    horns = [HORN_RANGE[0] + math.radians(horn_step_deg) * k
             for k in range(int(round(math.degrees(HORN_RANGE[1] - HORN_RANGE[0]) / horn_step_deg)) + 1)]
    los, his = [], []
    for t in thighs:
        best, cur = None, None
        for h in horns + [None]:
            ok = h is not None and drive_pose_allowed(p, t, h, mech)
            if ok:
                cur = (cur[0], h) if cur else (h, h)
            elif cur:
                if best is None or cur[1] - cur[0] > best[1] - best[0]:
                    best = cur
                cur = None
        los.append(best[0] if best else float("nan"))
        his.append(best[1] if best else float("nan"))
    return DriveStops(tuple(thighs), tuple(los), tuple(his))


# ---- pantograph leg (leg_drive 1) ---------------------------------------------------

def is_pantograph(p: dict[str, float]) -> bool:
    return float(p.get("leg_drive", 0.0)) >= 0.5


def _ang(ax: float, az: float, bx: float, bz: float) -> float:
    c = (ax * bx + az * bz) / (math.hypot(ax, az) * math.hypot(bx, bz) + 1e-12)
    return math.acos(max(-1.0, min(1.0, c)))


def panto_points(p: dict[str, float], horn_f: float, horn_r: float) -> dict | None:
    """Forward kinematics of the pantograph leg in the leg plane (x forward, z up,
    origin at the hip point; angles body-relative, 0 = straight down, + = backward):
    horn tips, the knee pin, the rod directions past the pin, the ankle, and the
    link angles (horn-rod at each tip, rod-rod at the pin). None when the rods
    cannot meet. The pin is the lower of the two circle intersections (the rods
    hang below the horns and cross)."""
    d, h, r, e, s = p["servo_dx"], p["horn_len"], p["rod_len"], p["rod_ext"], p["strut_len"]
    uf, ur = _u(horn_f), _u(horn_r)
    t1 = (d / 2 + h * uf[0], h * uf[1])
    t2 = (-d / 2 + h * ur[0], h * ur[1])
    dx, dz = t2[0] - t1[0], t2[1] - t1[1]
    dist = math.hypot(dx, dz)
    if dist < 1e-6 or dist > 2 * r - 1e-9:
        return None
    hh = math.sqrt(max(0.0, r * r - (dist / 2) ** 2))
    mx, mz = (t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2
    nx, nz = -dz / dist, dx / dist
    a1 = (mx + hh * nx, mz + hh * nz)
    a2 = (mx - hh * nx, mz - hh * nz)
    a = a1 if a1[1] < a2[1] else a2
    u1 = ((a[0] - t1[0]) / r, (a[1] - t1[1]) / r)
    u2 = ((a[0] - t2[0]) / r, (a[1] - t2[1]) / r)
    ankle = (a[0] + e * u1[0] + (e + s) * u2[0], a[1] + e * u1[1] + (e + s) * u2[1])
    return {"t1": t1, "t2": t2, "pin": a, "u1": u1, "u2": u2, "ankle": ankle,
            "rod_f": math.atan2(-u1[0], -u1[1]), "rod_r": math.atan2(-u2[0], -u2[1]),
            "horn_rod_f": _ang(uf[0], uf[1], u1[0], u1[1]), "horn_rod_r": _ang(ur[0], ur[1], u2[0], u2[1]),
            "cross": _ang(u1[0], u1[1], u2[0], u2[1])}


def panto_pose_allowed(p: dict[str, float], horn_f: float, horn_r: float) -> bool:
    """Hard-stop rule of the pantograph: horns within HORN_RANGE, the rods meet, and
    no pair of links (horn-rod at either tip, rod-rod at the pin) within FOLD_MARGIN of
    straight or folded."""
    if not (HORN_RANGE[0] - 1e-9 <= horn_f <= HORN_RANGE[1] + 1e-9
            and HORN_RANGE[0] - 1e-9 <= horn_r <= HORN_RANGE[1] + 1e-9):
        return False
    k = panto_points(p, horn_f, horn_r)
    if k is None:
        return False
    lo, hi = FOLD_MARGIN - 1e-9, math.pi - FOLD_MARGIN + 1e-9
    return all(lo <= k[n] <= hi for n in ("horn_rod_f", "horn_rod_r", "cross"))


def panto_ik(p: dict[str, float], x: float, z: float, guess: tuple[float, float] | None = None,
             tol: float = 1e-6) -> tuple[float, float, float]:
    """Horn angles (front, rear) that put the ankle at (x, z) relative to the hip
    point: damped Newton from `guess` (default: the horns splayed 60 deg). Returns
    (horn_f, horn_r, residual m); the residual is ~0 when the target is reachable
    with the links clear of their folds, else the nearest pose the search found."""
    hf, hr = guess if guess is not None else (-math.pi / 3, math.pi / 3)
    best = (hf, hr, float("inf"))
    lam = 1e-3

    def err(a, b):
        k = panto_points(p, a, b)
        if k is None:
            return None
        return (k["ankle"][0] - x, k["ankle"][1] - z)

    f = err(hf, hr)
    if f is None:
        for a, b in ((-1.0, 1.0), (-0.5, 0.5), (-1.5, 1.5), (0.0, 1.0), (-1.0, 0.0)):
            f = err(a, b)
            if f is not None:
                hf, hr = a, b
                break
        else:
            return best
    for _ in range(40):
        n = math.hypot(*f)
        if n < best[2] and panto_pose_allowed(p, hf, hr):
            best = (hf, hr, n)
        if n < tol:
            return best
        eps = 1e-6
        fa, fb = err(hf + eps, hr), err(hf, hr + eps)
        if fa is None or fb is None:
            break
        j = ((fa[0] - f[0]) / eps, (fb[0] - f[0]) / eps, (fa[1] - f[1]) / eps, (fb[1] - f[1]) / eps)
        # Levenberg-Marquardt step on J^T J + lam I
        g0, g1 = j[0] * f[0] + j[2] * f[1], j[1] * f[0] + j[3] * f[1]
        m00, m01, m11 = j[0] ** 2 + j[2] ** 2 + lam, j[0] * j[1] + j[2] * j[3], j[1] ** 2 + j[3] ** 2 + lam
        det = m00 * m11 - m01 * m01
        if abs(det) < 1e-18:
            break
        da, db = -(m11 * g0 - m01 * g1) / det, -(m00 * g1 - m01 * g0) / det
        step = math.hypot(da, db)
        if step > 0.5:
            da, db = da * 0.5 / step, db * 0.5 / step
        na = max(HORN_RANGE[0], min(HORN_RANGE[1], hf + da))
        nb = max(HORN_RANGE[0], min(HORN_RANGE[1], hr + db))
        fn = err(na, nb)
        if fn is None or math.hypot(*fn) > n:
            lam *= 10.0                                 # a worse step: damp harder, keep the pose
            if lam > 1e3:
                break
            continue
        lam = max(1e-6, lam / 3.0)
        hf, hr, f = na, nb, fn
    return best


def panto_reach(p: dict[str, float]) -> float:
    """Straight-down reach of the pantograph: the largest hip-to-ankle drop at x = 0
    the horns can set with every link clear of its folds (bisection)."""
    top = p["horn_len"] + p["rod_len"] + 2 * p["rod_ext"] + p["strut_len"]
    # the reachable drops are a band (the leg cannot fold flat either): walk down from the
    # longest link sum to the first reachable drop, then bisect up to the edge above it
    n, guess, lo, hi = 24, None, 0.0, top
    for k in range(1, n + 1):
        H = top * (1 - k / n) + 0.02 * k / n
        a, b, res = panto_ik(p, 0.0, -H, guess)
        if res < 1e-4:
            lo, hi, guess = H, top * (1 - (k - 1) / n) + 0.02 * (k - 1) / n, (a, b)
            break
    else:
        return 0.0
    for _ in range(14):
        mid = (lo + hi) / 2
        a, b, res = panto_ik(p, 0.0, -mid, guess)
        if res < 1e-4:
            lo, guess = mid, (a, b)
        else:
            hi = mid
    return lo


def leg_reach(p: dict[str, float]) -> float:
    """Hip-to-ankle distance at full extension: thigh + shin, or the pantograph's reach."""
    return panto_reach(p) if is_pantograph(p) else p["thigh_len"] + p["shin_len"]


def panto_nominal(p: dict[str, float]) -> tuple[float, float]:
    """Horn angles of the parked crouch: the ankle straight below the hip point at
    reach x cos(crouch / 2) (two equal links at knee flexion `crouch` stand at that
    height, so the gene means the same on either leg)."""
    reach = panto_reach(p)
    if reach <= 0.0:
        raise InvalidBody("pantograph rods cannot meet")
    a, b, res = panto_ik(p, 0.0, -reach * math.cos(p["crouch"] / 2))
    if res > 1e-4:
        raise InvalidBody("pantograph cannot park at the crouch")
    return a, b


def nominal_foot(p: dict[str, float]) -> tuple[float, float]:
    """Ankle position (x forward, z up) relative to the hip point in the parked crouch."""
    if is_pantograph(p):
        return panto_points(p, *panto_nominal(p))["ankle"]
    return foot_from_angles(p, *nominal_angles(p))


def roll_arm(p: dict[str, float]) -> float:
    """Roll pivot to leg plane: the bracket's length on the pantograph body, 0 on the four-bar (pivot at the leg plane)."""
    return max(0.0, p["hip_width"] / 2 - ROLL_PIVOT_Y) if is_pantograph(p) else 0.0


def roll_ratio(p: dict[str, float]) -> float:
    """Roll lever ratio, bracket arm over horn: derived on the pantograph, the gene on the four-bar."""
    return roll_arm(p) / p["horn_len"] if is_pantograph(p) else float(p.get("roll_ratio", 1.0))


def hip_roll_range(p: dict[str, float]) -> float:
    """The hip roll's half range: the roll servo's horn sweep through the lever, capped at HIP_ROLL_RANGE."""
    return min(HIP_ROLL_RANGE, HORN_RANGE[1] / max(1e-6, roll_ratio(p)))


def horn_stiffness(p: dict[str, float]) -> float:
    """A horn's flexural stiffness about its servo axis (N m/rad); all three horns are the same part."""
    return 10.0 ** float(p.get("horn_k_log10", 3.0))


def series_kp(kp_servo: float, k_horn: float) -> float:
    """Stiffness of the servo's position loop in series with its flexing horn."""
    return 1.0 / (1.0 / kp_servo + 1.0 / k_horn)


def nominal_height(p: dict[str, float]) -> float:
    """Hip-joint height above the floor in the parked crouch."""
    return -nominal_foot(p)[1] - foot_layout(p).bottom_z


def spawn_height(p: dict[str, float], terrain_zmax: float = 0.0) -> float:
    return nominal_height(p) + 0.01 + terrain_zmax


# ---- masses ---------------------------------------------------------------------

@dataclass(frozen=True)
class SkaterStructure:
    hip_kg: float               # shell + hardware + six servos
    core_kg: float              # shell + fan arms
    core_half_x: float
    core_half_y: float
    payload_kg: float
    payload_half: tuple[float, float, float]
    thigh_kg: float             # tube + parallelogram coupler
    shin_kg: float              # tube + bracket + tread drive (motor, worm, belt)
    foot_kg: float              # plate + rubber + rollers + ankle spring
    fan_kg: float               # each, incl. its pitch servo
    servo_kg: float             # each leg servo
    total_kg: float
    leg_fraction: float         # (thigh + shin + foot) / total, per leg


def structure_from_params(p: dict[str, float]) -> SkaterStructure:
    servo = servo_from_params(p)
    tread = tread_drive_from_params(p)
    fan = fan_from_params(p)
    lay = foot_layout(p)

    hip = (SHELL_KG_PER_M2 * _box_area(HIP_HALF_X, hip_half_y(p), HIP_HALF_Z)
           + HIP_HARDWARE_KG + 2 * (SERVOS_PER_LEG - 1) * servo.mass_kg + 2 * roll_servo_from_params(p).mass_kg)
    arm = math.sqrt(p["fan_x"] ** 2 + fan_center_y(p) ** 2 + fan_z_core(p) ** 2)
    cx, cy = core_half_x(p), core_half_y(p)
    core = SHELL_KG_PER_M2 * _box_area(cx, cy, p["core_height"] / 2) + 2 * FAN_ARM_KG_PER_M * arm
    battery = BATTERY_KG_BASE + BATTERY_KG_PER_MAH * p["battery_mah"]
    payload = battery + ELECTRONICS_KG

    if is_pantograph(p):
        # upper: two horns, two rods (tip to pin to the far node) and the pin; lower: the two
        # parallelogram links with the strut, their three pins, the ankle bracket, the tread drive
        thigh = 2 * HORN_KG + 2 * LINK_KG_PER_M * (p["rod_len"] + p["rod_ext"]) + PIN_KG
        shin = (LINK_KG_PER_M * (2 * p["rod_ext"] + p["strut_len"]) + 3 * PIN_KG + JOINT_BRACKET_KG
                + tread.mass_kg)
    else:
        thigh = (LINK_KG_PER_M + COUPLER_KG_PER_M) * p["thigh_len"] + JOINT_BRACKET_KG
        shin = LINK_KG_PER_M * p["shin_len"] + JOINT_BRACKET_KG + tread.mass_kg
    sole_area = p["sole_len"] * 2 * sole_half_w(p)
    n_rollers = len(lay.roller_x) + (1 if lay.idler else 0)
    foot = (SOLE_PLATE_KG_PER_M2 * sole_area + TREAD_RUBBER_KG_PER_M2 * 2 * sole_area
            + ROLLER_KG_PER_M * 2 * roller_half_len(p) * n_rollers + 2 * ANKLE_SPRING_KG)
    fan_kg = fan.mass_kg                            # fan_from_params already includes one pitch servo
    total = hip + core + payload + 2 * (thigh + shin + foot + fan_kg)
    return SkaterStructure(hip, core, cx, cy, payload, battery_half(p), thigh, shin, foot,
                           fan_kg, servo.mass_kg, total, (thigh + shin + foot) / total)


def leg_mass_fraction(p: dict[str, float]) -> float:
    return structure_from_params(p).leg_fraction


def check_buildable(p: dict[str, float]) -> None:
    """Raise InvalidBody for geometry that cannot be assembled."""
    foot_layout(p)
    if is_pantograph(p):
        hf, hr = panto_nominal(p)                           # raises when the rods cannot meet or park
        k = panto_points(p, hf, hr)
        # a step's worth of workspace: a 4 cm lift exactly, 6 cm fore and aft to within the
        # same reach clamp the two-link leg lives with (a near-straight crouch has little of either)
        for dx_, dz_, tol in ((0.0, 0.04, 1e-3), (0.06, 0.0, 0.03), (-0.06, 0.0, 0.03)):
            if panto_ik(p, k["ankle"][0] + dx_, k["ankle"][1] + dz_, (hf, hr))[2] > tol:
                raise InvalidBody("pantograph workspace too small around the crouch")
    if nominal_height(p) < 0.05:
        raise InvalidBody("crouch leaves the hip on the floor")
    if p["hip_width"] / 2 < sole_half_w(p) + 0.005:
        raise InvalidBody("hip too narrow for the soles")
    if core_half_y(p) < battery_half(p)[1]:
        raise InvalidBody("battery wider than the core")


def is_buildable(p: dict[str, float]) -> bool:
    try:
        check_buildable(p)
    except InvalidBody:
        return False
    return True
