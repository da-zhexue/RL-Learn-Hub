from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


GO2_XML: Path = Path(__file__).parent / "xmls" / "go2.xml"
JOINT_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{part}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for part in ("hip", "thigh", "calf")
)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    counters: dict[str, int] = {}
    foot_names = {"FL", "FR", "RL", "RR"}
    for g in spec.geoms:
        if g.name in foot_names or g.contype != 1 or g.conaffinity != 1:
            continue
        body_name = g.body.name
        index = counters.get(body_name, 0)
        counters[body_name] = index + 1
        g.name = f"{body_name}_collision_{index}"
    return spec


def get_go2_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.30),
            joint_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.9,
                ".*_calf_joint": -1.8,
            },
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                XmlActuatorCfg(target_names_expr=(name,)) for name in JOINT_NAMES
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
        spec_fn=get_spec,
    )
