from __future__ import annotations

import mujoco

from . import course


def course_spec_fn(variant: str, scale: float):
    """Add one go2_ppo course at the world origin.

    mjWarp replicates a single model into every world, so every world already
    has its own copy of this course. One course per environment would put all
    copies into every world (O(num_envs * ngeom) broadphase pairs) without
    giving worlds distinct courses.
    """

    boxes = tuple(course._kept_boxes(variant, scale))

    def apply(spec: mujoco.MjSpec) -> None:
        terrain_body = spec.body("terrain")
        for box_id, (x_lo, x_hi, y_lo, y_hi, top, bottom) in enumerate(boxes):
            terrain_body.add_geom(
                name=f"course_{box_id}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(
                    float(0.5 * (x_lo + x_hi)),
                    float(0.5 * (y_lo + y_hi)),
                    float(0.5 * (top + bottom)),
                ),
                size=(
                    float(0.5 * (x_hi - x_lo)),
                    float(0.5 * (y_hi - y_lo)),
                    float(0.5 * (top - bottom)),
                ),
            )

    return apply
