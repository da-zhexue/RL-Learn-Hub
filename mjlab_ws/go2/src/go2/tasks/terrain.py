from __future__ import annotations

import math

import mujoco
import numpy as np

from . import course


def _env_origins(num_envs: int, env_spacing: float) -> np.ndarray:
    rows = int(math.ceil(num_envs / int(math.sqrt(num_envs))))
    cols = int(math.ceil(num_envs / rows))
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    out = np.zeros((num_envs, 3), dtype=np.float64)
    out[:, 0] = -(ii.flatten()[:num_envs] - (rows - 1) / 2) * env_spacing
    out[:, 1] = (jj.flatten()[:num_envs] - (cols - 1) / 2) * env_spacing
    return out


def course_spec_fn(
    num_envs: int,
    env_spacing: float,
    variant: str,
    scale: float,
):
    """Add one go2_ppo course to the existing mjlab ground body."""

    origins = _env_origins(num_envs, env_spacing)
    boxes = tuple(course._kept_boxes(variant, scale))

    def apply(spec: mujoco.MjSpec) -> None:
        terrain_body = spec.body("terrain")
        for env_id, origin in enumerate(origins):
            for box_id, (x_lo, x_hi, y_lo, y_hi, top, bottom) in enumerate(boxes):
                terrain_body.add_geom(
                    name=f"course_{env_id}_{box_id}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=(
                        float(origin[0] + 0.5 * (x_lo + x_hi)),
                        float(origin[1] + 0.5 * (y_lo + y_hi)),
                        float(0.5 * (top + bottom)),
                    ),
                    size=(
                        float(0.5 * (x_hi - x_lo)),
                        float(0.5 * (y_hi - y_lo)),
                        float(0.5 * (top - bottom)),
                    ),
                )

    return apply
