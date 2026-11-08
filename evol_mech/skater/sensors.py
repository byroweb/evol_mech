"""Foot sensing the built skater will have, modelled so the controller never sees cleaner
data than the real parts give (docs/skater_test_plan.md, phase 0).

Two force-sensing resistors per foot, heel and toe, under the sole plate. In the model the
floor's contact forces on the foot's geoms are summed into the two bins by where they land
along the sole, then pass through an FSR-like response (nothing below a threshold, soft
saturation above a body weight or so), a first-order low-pass (the sensor's RC and the
ADC's filter), a 10-bit quantiser over 0-2 body weights, and a dropped reading now and
then. The controller gets, per foot:

  load     heel + toe, in body weights
  contact  load above a debounced threshold (on after 2 steps above, off after 2 below)
  cop      centre of pressure along the sole, metres ahead of the ankle (0 when unloaded)

The passive ankle's angle is read as well (a hall sensor across the joint on the build): a
second, independent load estimate. Servo load is reported from the actuator force as a
fraction of stall, filtered with the servo's electrical lag; it is reported, not yet used.

`summary()` gives the rollout the per-step numbers the plan asks for: load at touchdown,
the centre of pressure's excursion along the sole per stance, the stance fraction measured
from the sensor rather than the clock, and how many readings were dropped.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

FSR_THRESHOLD_BW = 0.01      # nothing registers below this (body weights)
FSR_SAT_BW = 1.5             # soft saturation: reads tanh(f / sat) * sat
FSR_TAU_S = 0.008            # sensor + ADC filter time constant
FSR_ADC_BITS = 10            # over 0-2 body weights
FSR_FULL_SCALE_BW = 2.0
FSR_POS_FRAC = 0.8           # heel and toe pads at +-0.8 of the half sole length from the ankle
CONTACT_ON_BW = 0.05         # debounced contact: on above this for CONTACT_STEPS, off below CONTACT_OFF_BW for the same
CONTACT_OFF_BW = 0.02
CONTACT_STEPS = 2
DROPOUT_PER_STEP = 0.002     # a channel reads zero this often (about one dropped reading per 5 s per channel at 100 Hz)
TOUCHDOWN_WINDOW_S = 0.05    # the impact is the load's peak this long after a contact rise
SERVO_TAU_S = 0.02           # electrical lag of a current reading


class FootSensors:
    def __init__(self, model: mujoco.MjModel, p: dict[str, float], total_mass: float, dt: float,
                 rng: np.random.Generator | None = None, dropout: float = DROPOUT_PER_STEP):
        self.model, self.dt, self.dropout = model, dt, dropout
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.weight = total_mass * 9.81
        self.pad_x = FSR_POS_FRAC * p["sole_len"] / 2
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.side_of: dict[int, str] = {}
        for g in range(model.ngeom):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if n.startswith("foot_"):
                self.side_of[g] = n[5]
        self.foot_body = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{s}") for s in "lr"}
        self.ankle_q = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ankle_{s}")] for s in "lr"}
        self.servo_acts = [i for i in range(model.nu)
                           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith(("hip_roll", "hip_pitch", "knee"))]
        self.servo_stall = np.array([model.actuator_forcerange[i][1] for i in self.servo_acts])
        self.alpha = 1.0 - math.exp(-dt / FSR_TAU_S)
        self.servo_alpha = 1.0 - math.exp(-dt / SERVO_TAU_S)
        self._f6 = np.zeros(6)
        # state per foot: filtered heel / toe (body weights), debounce counters, contact flag
        self.filt = {s: np.zeros(2) for s in "lr"}
        self.count = {s: 0 for s in "lr"}
        self.contact = {s: False for s in "lr"}
        self.load = {s: 0.0 for s in "lr"}
        self.cop = {s: 0.0 for s in "lr"}
        self.ankle = {s: 0.0 for s in "lr"}
        self.servo_load = np.zeros(len(self.servo_acts))
        # bookkeeping for summary()
        self.steps = 0
        self.stance_steps = {s: 0 for s in "lr"}
        self.dropped = 0
        self._since_rise = {s: None for s in "lr"}       # steps since the contact rise, while in the touchdown window
        self._peak = {s: 0.0 for s in "lr"}
        self.touchdown_loads: list[float] = []
        self._cop_lo = {s: math.inf for s in "lr"}
        self._cop_hi = {s: -math.inf for s in "lr"}
        self.cop_excursions: list[float] = []

    def _raw(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Heel and toe normal force per foot (N) from this step's floor contacts."""
        raw = {s: np.zeros(2) for s in "lr"}
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.floor and g2 in self.side_of:
                g = g2
            elif g2 == self.floor and g1 in self.side_of:
                g = g1
            else:
                continue
            s = self.side_of[g]
            mujoco.mj_contactForce(self.model, data, ci, self._f6)
            fn = abs(float(self._f6[0]))
            b = self.foot_body[s]
            local = data.xmat[b].reshape(3, 3).T @ (np.asarray(c.pos) - data.xpos[b])
            # split between the pads by the lever: a force at x lands (pad_x + x) / (2 pad_x) on the toe pad
            w_toe = min(1.0, max(0.0, (self.pad_x + float(local[0])) / (2 * self.pad_x)))
            raw[s] += fn * np.array([1.0 - w_toe, w_toe])
        return raw

    def update(self, data: mujoco.MjData) -> None:
        self.steps += 1
        raw = self._raw(data)
        q = FSR_FULL_SCALE_BW / (2 ** FSR_ADC_BITS - 1)
        for s in "lr":
            bw = raw[s] / self.weight
            bw = np.where(bw < FSR_THRESHOLD_BW, 0.0, FSR_SAT_BW * np.tanh(bw / FSR_SAT_BW))
            self.filt[s] += self.alpha * (bw - self.filt[s])
            read = np.round(np.clip(self.filt[s], 0.0, FSR_FULL_SCALE_BW) / q) * q
            if self.dropout > 0.0:
                drop = self.rng.random(2) < self.dropout
                if drop.any():
                    read = np.where(drop, 0.0, read)
                    self.dropped += int(drop.sum())
            heel, toe = float(read[0]), float(read[1])
            load = heel + toe
            self.load[s] = load
            self.cop[s] = self.pad_x * (toe - heel) / load if load > 0.0 else 0.0
            self.ankle[s] = float(data.qpos[self.ankle_q[s]])
            # debounce
            if self.contact[s]:
                self.count[s] = self.count[s] + 1 if load < CONTACT_OFF_BW else 0
                if self.count[s] >= CONTACT_STEPS:
                    self.contact[s], self.count[s] = False, 0
                    if math.isfinite(self._cop_lo[s]):
                        self.cop_excursions.append(self._cop_hi[s] - self._cop_lo[s])
                    self._cop_lo[s], self._cop_hi[s] = math.inf, -math.inf
            else:
                self.count[s] = self.count[s] + 1 if load > CONTACT_ON_BW else 0
                if self.count[s] >= CONTACT_STEPS:
                    self.contact[s], self.count[s] = True, 0
                    self._since_rise[s], self._peak[s] = 0, load
            if self.contact[s]:
                self.stance_steps[s] += 1
                self._cop_lo[s] = min(self._cop_lo[s], self.cop[s])
                self._cop_hi[s] = max(self._cop_hi[s], self.cop[s])
            if self._since_rise[s] is not None:
                self._peak[s] = max(self._peak[s], load)
                self._since_rise[s] += 1
                if self._since_rise[s] * self.dt >= TOUCHDOWN_WINDOW_S:
                    self.touchdown_loads.append(self._peak[s])
                    self._since_rise[s] = None
        if self.servo_acts:
            frac = np.abs(data.actuator_force[self.servo_acts]) / self.servo_stall
            self.servo_load += self.servo_alpha * (frac - self.servo_load)

    def any_contact(self) -> bool:
        return self.contact["l"] or self.contact["r"]

    def summary(self) -> dict[str, float]:
        n = max(1, self.steps)
        return {
            "sensor_stance_frac": 0.5 * (self.stance_steps["l"] + self.stance_steps["r"]) / n,
            "touchdown_load_bw": float(np.mean(self.touchdown_loads)) if self.touchdown_loads else 0.0,
            "touchdown_load_max_bw": float(max(self.touchdown_loads)) if self.touchdown_loads else 0.0,
            "cop_excursion_m": float(np.mean(self.cop_excursions)) if self.cop_excursions else 0.0,
            "sensor_dropouts": float(self.dropped),
            "servo_load_peak": float(self.servo_load.max()) if self.servo_acts else 0.0,
        }
