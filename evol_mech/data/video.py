"""Offscreen rendering of a rollout to MP4. Only ever call this for elites:
rendering is 50-100x slower than simulating."""
from __future__ import annotations

from pathlib import Path

import imageio
import mujoco
import numpy as np

from ..bodies import get_body

FPS = 30
WIDTH, HEIGHT = 640, 480


def render_rollout(genome: np.ndarray, task: dict, out_path: Path | str, seed: int = 0,
                   fans_enabled: bool = True, body: str = "biped") -> dict:
    frames: list[np.ndarray] = []
    state = {"renderer": None, "next_t": 0.0}
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.distance = 1.2
    cam.elevation = -15
    cam.azimuth = 135

    def cb(model, data, t):
        if state["renderer"] is None:
            state["renderer"] = mujoco.Renderer(model, HEIGHT, WIDTH)
            cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
        if t >= state["next_t"]:
            state["renderer"].update_scene(data, cam)
            frames.append(state["renderer"].render().copy())
            state["next_t"] += 1.0 / FPS

    result = get_body(body).rollout(genome, task, seed=seed, fans_enabled=fans_enabled, frame_cb=cb)
    if state["renderer"] is not None:
        state["renderer"].close()
    if frames:
        imageio.mimwrite(str(out_path), frames, fps=FPS, codec="libx264", quality=7)
    return result
