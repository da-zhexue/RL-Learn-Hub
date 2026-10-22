from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch

from . import core
from .config import EnvCfg, RewardCfg


LEGS = ("FR", "FL", "RR", "RL")
PARTS = ("hip", "thigh", "calf")
JOINT_NAMES: tuple[str, ...] = tuple(f"{leg}_{part}_joint" for leg in LEGS for part in PARTS)
ACTUATOR_NAMES: tuple[str, ...] = tuple(f"{leg}_{part}" for leg in LEGS for part in PARTS)
FOOT_GEOM_NAMES: tuple[str, ...] = ("FR", "FL", "RR", "RL")


@dataclass
class Ctx:
    cfg: EnvCfg
    reward_cfg: RewardCfg
    terrain: core.CourseTerrain
    joint_ids: list[int]
    actuator_ids: list[int]
    foot_geom_ids: list[int]
    penalty_geom_ids: list[int]
    foot_radius: torch.Tensor
    goal_z: float
    clearance_target: float


def setup(env) -> Ctx:
    cfg = EnvCfg(
        terrain=env.cfg.terrain_variant,
        terrain_scale=env.cfg.terrain_scale,
        privileged=env.cfg.privileged,
        height_scan=env.cfg.height_scan,
    )
    robot = env.scene["robot"]
    mj_model: mujoco.MjModel = env.sim.mj_model

    joint_ids, joint_names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    if tuple(joint_names) != JOINT_NAMES:
        raise RuntimeError(f"joint names mismatch: requested={JOINT_NAMES}, got={tuple(joint_names)}")
    if len(joint_ids) != 12:
        raise RuntimeError(f"expected 12 joints, found {len(joint_ids)}")

    actuator_ids, actuator_names = robot.find_actuators(list(ACTUATOR_NAMES), preserve_order=True)
    if tuple(actuator_names) != ACTUATOR_NAMES:
        raise RuntimeError(f"actuator names mismatch: requested={ACTUATOR_NAMES}, got={tuple(actuator_names)}")

    foot_ids, foot_geom_names = robot.find_geoms(list(FOOT_GEOM_NAMES), preserve_order=True)
    if tuple(foot_geom_names) != FOOT_GEOM_NAMES:
        raise RuntimeError(f"foot geom names mismatch: requested={FOOT_GEOM_NAMES}, got={tuple(foot_geom_names)}")

    collision_ids, collision_geom_names = robot.find_geoms((".*_collision.*",), preserve_order=True)
    keep = [i for i, n in zip(collision_ids, collision_geom_names) if n not in set(FOOT_GEOM_NAMES)]

    global_foot_ids = robot.indexing.geom_ids[torch.tensor(foot_ids, device=env.device)].tolist()
    global_penalty_ids = robot.indexing.geom_ids[torch.tensor(keep, device=env.device)].tolist()
    foot_radius = torch.as_tensor(
        mj_model.geom_size[global_foot_ids, 0],
        dtype=torch.float64,
        device=env.device,
    )
    reward_cfg = RewardCfg()
    terrain = core.CourseTerrain(cfg.terrain, cfg.terrain_scale, device=env.device, dtype=torch.float64)
    tops = terrain.stair_tops
    rise = max((b - a for a, b in zip(tops, tops[1:])), default=0.0)
    clearance_target = (
        max(float(reward_cfg.clearance_target), rise + reward_cfg.clearance_margin)
        if tops
        else float(reward_cfg.clearance_target)
    )

    return Ctx(
        cfg=cfg,
        reward_cfg=reward_cfg,
        terrain=terrain,
        joint_ids=list(joint_ids),
        actuator_ids=list(actuator_ids),
        foot_geom_ids=list(foot_ids),
        penalty_geom_ids=keep,
        foot_radius=foot_radius,
        goal_z=core.goal_z_for(cfg, terrain),
        clearance_target=clearance_target,
    )


def _ctx(env) -> Ctx:
    got = getattr(env, "_go2_ctx", None)
    if got is None:
        got = setup(env)
        env._go2_ctx = got
    return got
