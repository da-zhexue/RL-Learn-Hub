from __future__ import annotations

import torch

from mjlab.envs.mdp.events import resolve_env_ids

from . import core
from .context import Ctx, _ctx
from .context import FOOT_GEOM_NAMES

F64 = torch.float64


def _u(lo: float, hi: float, n: int, device: str) -> torch.Tensor:
    return torch.empty(n, dtype=F64, device=device).uniform_(lo, hi)


def _course_xy(env, pos_w: torch.Tensor) -> torch.Tensor:
    return pos_w[:, :2].to(F64) - env.scene.env_origins[:, :2].to(F64)


def _default_prev(env) -> dict:
    n = env.num_envs
    return {
        "prev_action": torch.zeros(n, 12, dtype=F64, device=env.device),
        "prev_x": torch.zeros(n, dtype=F64, device=env.device),
        "prev_terrain_h": torch.zeros(n, dtype=F64, device=env.device),
        "prev_terrain_level": torch.zeros(n, dtype=torch.long, device=env.device),
        "prev_foot_x": None,
        "foot_air_time": torch.zeros(n, 4, dtype=F64, device=env.device),
        "foot_clear_ema": torch.zeros(n, 4, dtype=F64, device=env.device),
        "fresh": torch.ones(n, dtype=torch.bool, device=env.device),
    }


def reset_prev(env, env_ids=None) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    prev = getattr(env, "_go2_prev", None)
    if prev is None:
        env._go2_prev = _default_prev(env)
        prev = env._go2_prev
    prev["fresh"][env_ids] = True
    prev["foot_air_time"][env_ids] = 0.0
    prev["foot_clear_ema"][env_ids] = 0.0


def reset_base(env, env_ids=None) -> None:
    """Match go2_ppo's reset distribution, including footprint-aware z."""

    env_ids = resolve_env_ids(env, env_ids)
    ctx = _ctx(env)
    robot = env.scene["robot"]
    n = len(env_ids)

    x = _u(ctx.cfg.reset_x[0], ctx.cfg.reset_x[1], n, env.device)
    stair = torch.rand(n, device=env.device) < ctx.cfg.reset_frac_stair
    if bool(stair.any()):
        stair_x = _u(
            ctx.cfg.reset_x_stair[0],
            ctx.cfg.reset_x_stair[1],
            int(stair.sum()),
            env.device,
        )
        x[stair] = stair_x
    y = torch.empty(n, dtype=F64, device=env.device).uniform_(
        -ctx.cfg.reset_y, ctx.cfg.reset_y
    )
    yaw = torch.empty(n, dtype=F64, device=env.device).uniform_(
        -ctx.cfg.reset_yaw, ctx.cfg.reset_yaw
    )
    z = _u(ctx.cfg.reset_z[0], ctx.cfg.reset_z[1], n, env.device)
    z += ctx.terrain.top_footprint(x, y)

    quat = torch.zeros(n, 4, dtype=F64, device=env.device)
    quat[:, 0] = torch.cos(yaw / 2.0)
    quat[:, 3] = torch.sin(yaw / 2.0)
    pos = torch.stack(
        [x + env.scene.env_origins[env_ids, 0], y + env.scene.env_origins[env_ids, 1], z],
        dim=-1,
    )
    root_state = torch.cat(
        [
            pos,
            quat,
            torch.zeros(n, 6, dtype=F64, device=env.device),
        ],
        dim=-1,
    )
    robot.write_root_state_to_sim(root_state.to(torch.float32), env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids].clone(),
        torch.zeros_like(robot.data.default_joint_vel[env_ids]),
        env_ids=env_ids,
    )


def _foot_state(env, ctx: Ctx, prev: dict):
    robot = env.scene["robot"]
    pos_w = robot.data.geom_pos_w[:, ctx.foot_geom_ids].to(F64)
    xy = pos_w[..., :2] - env.scene.env_origins[:, None, :2].to(F64)
    ground = ctx.terrain.top(xy[..., 0], xy[..., 1])
    clearance = pos_w[..., 2] - ctx.foot_radius - ground

    sensor = env.scene["feet_ground_contact"]
    found = sensor.data.found
    if found is None:
        raise RuntimeError("feet_ground_contact must provide `found`")
    names = sensor.primary_names
    perm = [names.index(name) for name in FOOT_GEOM_NAMES]
    found = found[:, perm]
    air = found <= 0

    if prev["prev_foot_x"] is None:
        vx = torch.zeros_like(clearance)
    else:
        vx = (pos_w[..., 0] - prev["prev_foot_x"]) / env.step_dt
    return clearance, vx, air, pos_w[..., 0].clone()


def _contact_force(env) -> torch.Tensor:
    sensor = env.scene["body_ground_contact"]
    force = sensor.data.force
    if force is None:
        raise RuntimeError("body_ground_contact must provide `force`")
    return torch.linalg.vector_norm(force.to(F64), dim=-1).sum(dim=-1)


def _height_scan(env, ctx: Ctx, base_pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    if not ctx.cfg.height_scan:
        return base_pos.new_zeros((env.num_envs, 0))
    offsets = torch.tensor(
        [(dx, dy, 0.0) for dx in ctx.cfg.scan_x for dy in ctx.cfg.scan_y],
        dtype=F64,
        device=env.device,
    )
    w, x, y, z = quat.unbind(-1)
    qv = torch.stack([x, y, z], dim=-1)
    t = 2.0 * torch.cross(qv[:, None, :], offsets[None, :, :], dim=-1)
    rotated = offsets[None, :, :] + w[:, None, None] * t + torch.cross(
        qv[:, None, :], t, dim=-1
    )
    sample_xy = base_pos[:, None, :2] + rotated[..., :2]
    ground = ctx.terrain.top(sample_xy[..., 0], sample_xy[..., 1])
    return torch.clamp(ground - base_pos[:, 2:3], -1.0, 1.0)


def get_state(env) -> core.State:
    ctx = _ctx(env)
    robot = env.scene["robot"]
    prev = getattr(env, "_go2_prev", None)
    if prev is None:
        prev = _default_prev(env)
        env._go2_prev = prev

    pos_w = robot.data.root_link_pos_w.to(F64)
    base_pos = torch.cat([_course_xy(env, pos_w), pos_w[:, 2:3]], dim=-1)
    quat = robot.data.root_link_quat_w.to(F64)
    foot_clearance, foot_vx, foot_air, foot_x = _foot_state(env, ctx, prev)
    foot_xy = (
        robot.data.geom_pos_w[:, ctx.foot_geom_ids, :2].to(F64)
        - env.scene.env_origins[:, None, :2].to(F64)
    )
    terrain_h = ctx.terrain.support_height(foot_xy)
    terrain_level = ctx.terrain.level_at_height(terrain_h)

    if prev["fresh"].any():
        fresh = prev["fresh"]
        prev["prev_x"][fresh] = base_pos[fresh, 0]
        prev["prev_terrain_h"][fresh] = terrain_h[fresh]
        prev["prev_terrain_level"][fresh] = terrain_level[fresh]
        prev["prev_action"][fresh] = 0.0
        if prev["prev_foot_x"] is None:
            prev["prev_foot_x"] = foot_x.clone()
        else:
            prev["prev_foot_x"][fresh] = foot_x[fresh]
        prev["fresh"][fresh] = False

    landed_air_time = torch.where(
        (~foot_air) & (prev["foot_air_time"] > 0.0),
        prev["foot_air_time"],
        torch.zeros_like(prev["foot_air_time"]),
    )
    current_air_time = torch.where(
        foot_air,
        prev["foot_air_time"] + env.step_dt,
        torch.zeros_like(prev["foot_air_time"]),
    )
    clear_ema = (
        ctx.reward_cfg.clearance_ema_alpha * prev["foot_clear_ema"]
        + (1.0 - ctx.reward_cfg.clearance_ema_alpha)
        * torch.clamp(foot_clearance, 0.0, 0.2)
    )

    cmd_vx = env.command_manager.get_command("course_velocity")[:, 0].to(F64)
    action = torch.clamp(
        env.action_manager.get_term("joints").raw_action.to(F64), -1.0, 1.0
    )
    st = core.State(
        base_pos=base_pos,
        base_quat=quat,
        base_lin_vel=robot.data.root_link_lin_vel_w.to(F64),
        base_ang_vel=robot.data.root_link_ang_vel_b.to(F64),
        # 用带编码器零位偏差的读数：dr.encoder_bias 写的是 encoder_bias，
        # 只有 joint_pos_biased 会把它加进来（策略观测和 PD 控制都该看到偏差）。
        q=robot.data.joint_pos_biased[:, ctx.joint_ids].to(F64),
        dq=robot.data.joint_vel[:, ctx.joint_ids].to(F64),
        tau=robot.data.actuator_force[:, ctx.actuator_ids].to(F64),
        contact_force=_contact_force(env),
        action=action,
        terrain_h=terrain_h,
        terrain_level=terrain_level,
        prev_x=prev["prev_x"],
        prev_action=prev["prev_action"],
        prev_terrain_h=prev["prev_terrain_h"],
        prev_terrain_level=prev["prev_terrain_level"],
        cmd_vx=cmd_vx,
        foot_clearance=foot_clearance,
        foot_vx=foot_vx,
        foot_air=foot_air,
        foot_air_time=landed_air_time,
        foot_clearance_ema=clear_ema,
        clearance_target=torch.full_like(foot_clearance, ctx.clearance_target),
        height_scan=_height_scan(env, ctx, base_pos, quat),
    )
    env._go2_step_aux = {
        "current_air_time": current_air_time,
        "clear_ema": clear_ema,
        "foot_x": foot_x,
    }
    return st


def advance_prev(env, env_ids=None) -> None:
    """Run as a `step` event, after reward/termination have read the old state."""

    st = get_state(env)
    prev = env._go2_prev
    aux = env._go2_step_aux
    prev["prev_action"] = st.action
    prev["prev_x"] = st.base_pos[:, 0]
    prev["prev_terrain_h"] = st.terrain_h
    prev["prev_terrain_level"] = torch.maximum(prev["prev_terrain_level"], st.terrain_level)
    prev["foot_air_time"] = aux["current_air_time"]
    prev["foot_clear_ema"] = aux["clear_ema"]
    prev["prev_foot_x"] = aux["foot_x"]
