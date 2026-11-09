"""The core's downward-looking multizone time-of-flight sensor: an ST VL53L8CX, modelled from its
datasheet (DS14161 rev 12) so the controller never sees cleaner terrain data than the real part gives.
The module only: supply, clock and bus are the carrier board's business.

What the part does, and how each property is modelled:

  zones        8 x 8 over a 45 x 45 deg square detection volume (datasheet table 2), 5.625 deg a zone.
               Each zone is cast as a 3 x 3 bundle of rays across its square.
  one number   A zone returns one distance per frame. Surfaces closer together than the part can
  per zone     separate (about 60 cm along the beam) blend into one signal-weighted mean, so a zone
               half on a foot and half on the floor reads between the two; farther apart, the
               strongest return wins (the driver's default target order). Signal per ray ~ cos(incidence)/d^2.
  range        2 cm to 4 m (table 1). A ray's reach shrinks with grazing incidence as sqrt(cos);
               `max_range_m` 1.1 models 5 klx daylight on a dark floor (table 19).
  frame rate   15 Hz in 8 x 8 (the 60 Hz of table 1 is 4 x 4 only). A frame is the scene at the middle
               of its integration, delivered at its end: about 33 ms old when the controller reads it.
  accuracy     +-11 mm below 20 cm, +-5 % beyond (table 20, 8 x 8 at 15 Hz), quoted at 2.7 sigma:
               Gaussian noise with sigma = that / 2.7, fresh every zone and frame.
  dropouts     A zone is flagged invalid (status 255, distance NaN) when under a quarter of its rays
               return, and otherwise at random `invalid_rate` of the time (the part's detection rate is
               90 % at the edge of its range; well inside it, a few percent).
  mounting     The datasheet adds 1-2 % for PCB tilt and housing: the module is tilted by up to
               `mount_error_deg` about both axes, drawn once per run, which the controller does not know.

Orientation. The grid here is in scene order, as the robot looks: row 0 is the far edge of the field
(ahead), row 7 the near edge (toward the feet); column 0 is the robot's left. The real part's lens
flips the image both ways (section 8.1.3): zone ID 0 sees the top right of the scene. `zone_ids()`
converts a scene-order grid into the driver's zone order and back.

The sensor's axis is the `lidar` site's z axis (mjcf._lidar). The robot's own legs and feet are in the
scene and occlude the floor, as they will on the build; only the core, the body the module is on, is excluded.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

ZONES = 8
FOV_DEG = 45.0               # detection volume, horizontal and vertical (datasheet table 2)
RATE_HZ = 15.0               # 8 x 8 ranging frequency
MIN_RANGE_M = 0.02
MAX_RANGE_M = 4.0            # dark, light targets, inner zones (table 19)
SUB = 3                      # rays per zone along each axis
SEPARATION_M = 0.6           # two surfaces closer than this along the beam merge into one return
NEAR_SIGMA_M = 0.011 / 2.7   # +-11 mm at 2.7 sigma, 2-20 cm
FAR_SIGMA_FRAC = 0.05 / 2.7  # +-5 % at 2.7 sigma, 20-400 cm
INVALID_RATE = 0.02
MOUNT_ERROR_DEG = 1.0
STATUS_VALID, STATUS_NONE = 5, 255     # the driver's target_status for a valid range and for no target

ZONE_RAD = math.radians(FOV_DEG / ZONES)


def zone_ids(grid: np.ndarray) -> np.ndarray:
    """Scene-order 8 x 8 grid -> the driver's 64 zones (row-major from zone 0), or 64 zones -> the grid:
    the lens flips both axes, so the same flip converts either way."""
    g = np.asarray(grid)
    return g[::-1, ::-1].reshape(-1) if g.ndim == 2 else g.reshape(ZONES, ZONES)[::-1, ::-1]


def _angles(r: float, c: float) -> tuple[float, float]:
    """(toward the far edge, toward the left) of a point at fractional row r, column c of the grid."""
    return (ZONES / 2 - r) * ZONE_RAD, (ZONES / 2 - c) * ZONE_RAD


def _dir(far: float, left: float) -> np.ndarray:
    v = np.array([-math.tan(far), math.tan(left), 1.0])
    return v / np.linalg.norm(v)


def ray_directions() -> np.ndarray:
    """Unit ray directions in the sensor frame, (8, 8, SUB*SUB, 3). Sensor frame: z along the axis,
    +y toward the robot's left, -x toward the far edge of the field."""
    out = np.zeros((ZONES, ZONES, SUB * SUB, 3))
    for r in range(ZONES):
        for c in range(ZONES):
            for i in range(SUB):
                for j in range(SUB):
                    out[r, c, i * SUB + j] = _dir(*_angles(r + (i + 0.5) / SUB, c + (j + 0.5) / SUB))
    return out


def zone_centres() -> np.ndarray:
    """Unit direction of each zone's centre in the sensor frame, (8, 8, 3)."""
    return np.array([[_dir(*_angles(r + 0.5, c + 0.5)) for c in range(ZONES)] for r in range(ZONES)])


class VL53L8CX:
    def __init__(self, model: mujoco.MjModel, rng: np.random.Generator | None = None, *,
                 site: str = "lidar", rate_hz: float = RATE_HZ, max_range_m: float = MAX_RANGE_M,
                 invalid_rate: float = INVALID_RATE, mount_error_deg: float = MOUNT_ERROR_DEG):
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            raise ValueError(f"no site {site!r}: build the model with the task's lidar key on")
        self.body = int(model.site_bodyid[self.site])
        self.period = 1.0 / rate_hz
        self.max_range, self.invalid_rate = max_range_m, invalid_rate
        ex, ey = np.radians(self.rng.uniform(-1, 1, 2) * mount_error_deg)
        cx, sx, cy, sy = math.cos(ex), math.sin(ex), math.cos(ey), math.sin(ey)
        self.mount = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        self.dirs = ray_directions().reshape(-1, 3) @ self.mount.T   # true sensor-frame rays, the tilt error included
        self.centres = zone_centres()                                # what the controller believes (no tilt error)
        n = len(self.dirs)
        self._vec = np.zeros(3 * n)
        self._gid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n)
        self._nrm = np.zeros(3 * n)
        # the delivered frame
        self.dist = np.full((ZONES, ZONES), np.nan)                # radial distance per zone, m (NaN invalid)
        self.status = np.full((ZONES, ZONES), STATUS_NONE, dtype=np.int16)
        self.frame_t = -1.0                                         # sample time of the delivered frame
        self.hits_world = np.full((ZONES, ZONES, 3), np.nan)        # where each zone's centre ray truly hit (views)
        self.origin_world = np.zeros(3)                             # the module's position at that sample
        self.rot_world = np.eye(3)                                  # and its orientation (sensor -> world)
        self.new_frame = False
        self.frames = 0
        self.valid_zones = 0
        self._next_sample = self.period / 2
        self._pending = None

    # ---- per control step -----------------------------------------------------------
    def update(self, data: mujoco.MjData, t: float) -> bool:
        """Call every control step. Returns True when a new frame was delivered this step."""
        self.new_frame = False
        if self._pending is not None and t >= self._pending[0]:
            _, self.frame_t, self.dist, self.status, self.hits_world, self.origin_world, self.rot_world = self._pending
            self._pending = None
            self.new_frame = True
            self.frames += 1
            self.valid_zones += int(np.sum(self.status == STATUS_VALID))
        if t >= self._next_sample:
            self._pending = (self._next_sample + self.period / 2, t, *self._measure(data))
            self._next_sample += self.period
        return self.new_frame

    def _measure(self, data: mujoco.MjData):
        R = data.site_xmat[self.site].reshape(3, 3).copy()
        p0 = data.site_xpos[self.site].copy()
        world = self.dirs @ R.T
        n = len(world)
        self._vec[:] = world.ravel()
        mujoco.mj_multiRay(self.model, data, p0, self._vec, None, 1, self.body, self._gid, self._dist, self._nrm,
                           n, self.max_range)
        cos = np.clip(-np.sum(self._nrm.reshape(-1, 3) * world, axis=1), 0.0, 1.0)
        ok = (self._dist >= MIN_RANGE_M) & (self._dist <= self.max_range * np.sqrt(cos)) & (self._gid >= 0)
        d = self._dist.reshape(ZONES, ZONES, -1)
        ok = ok.reshape(ZONES, ZONES, -1)
        w = np.where(ok, cos.reshape(ZONES, ZONES, -1) / np.maximum(d, 1e-3) ** 2, 0.0)
        dist = np.full((ZONES, ZONES), np.nan)
        status = np.full((ZONES, ZONES), STATUS_NONE, dtype=np.int16)
        for r in range(ZONES):
            for c in range(ZONES):
                m = ok[r, c]
                if m.sum() < 2:                    # under a quarter of the zone returns: too weak a signal
                    continue
                dd, ww = d[r, c][m], w[r, c][m]
                o = np.argsort(dd)
                dd, ww = dd[o], ww[o]
                gaps = np.where(np.diff(dd) > SEPARATION_M)[0]
                if len(gaps):                      # separable targets: the strongest return wins
                    g = max(np.split(np.arange(len(dd)), gaps + 1), key=lambda ix: ww[ix].sum())
                    dd, ww = dd[g], ww[g]
                dist[r, c] = float(np.sum(dd * ww) / np.sum(ww))
                status[r, c] = STATUS_VALID
        sigma = np.where(dist < 0.2, NEAR_SIGMA_M, FAR_SIGMA_FRAC * dist)
        dist = dist + self.rng.normal(0.0, 1.0, dist.shape) * np.nan_to_num(sigma)
        status[self.rng.random(dist.shape) < self.invalid_rate] = STATUS_NONE
        dist[status != STATUS_VALID] = np.nan
        # the true hit of each zone's centre ray (a view aid; the bundle's middle ray, SUB being odd)
        mid = (SUB * SUB) // 2
        cw = world.reshape(ZONES, ZONES, -1, 3)[:, :, mid]
        dm = d[:, :, mid]
        hits = np.where((dm > 0)[..., None], p0 + cw * dm[..., None], np.nan)
        return dist, status, hits, p0, R

    # ---- what the controller can make of a frame ---------------------------------------
    def points_sensor(self) -> np.ndarray:
        """The delivered frame as points in the sensor frame, (8, 8, 3), NaN where invalid: each zone's
        distance along its nominal centre direction (the mounting error is unknown to the robot)."""
        return self.centres * self.dist[..., None]

    def points_world(self) -> np.ndarray:
        """The delivered frame placed with the module's true pose at its sample, (8, 8, 3): what a perfect pose
        estimate would make of it (the mounting error and the range noise remain)."""
        return self.origin_world + self.points_sensor() @ self.rot_world.T

    def summary(self) -> dict:
        return {"lidar_frames": self.frames,
                "lidar_valid_frac": self.valid_zones / max(1, self.frames * ZONES * ZONES)}
