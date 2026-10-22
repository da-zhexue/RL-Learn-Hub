from __future__ import annotations

from . import core
from .context import _ctx
from .state import get_state


def _r(env):
    ctx = _ctx(env)
    return ctx.reward_cfg, get_state(env), ctx


def reward_progress(env):
    cfg, st, _ = _r(env)
    return core.term_progress(st, cfg)


def reward_track_lin_vel(env):
    cfg, st, _ = _r(env)
    return core.term_track_lin_vel(st, cfg)


def reward_climb(env):
    cfg, st, _ = _r(env)
    return core.term_climb(st, cfg)


def reward_level_bonus(env):
    cfg, st, _ = _r(env)
    return core.term_level_bonus(st, cfg)


def reward_feet_clearance(env):
    cfg, st, _ = _r(env)
    return core.term_feet_clearance(st, cfg)


def reward_feet_air_time(env):
    cfg, st, _ = _r(env)
    return core.term_feet_air_time(st, cfg)


def reward_foot_swing_height(env):
    cfg, st, _ = _r(env)
    return core.term_foot_swing_height(st, cfg)


def reward_base_height(env):
    cfg, st, _ = _r(env)
    return core.term_base_height(st, cfg)


def reward_lateral(env):
    cfg, st, _ = _r(env)
    return core.term_lateral(st, cfg)


def reward_lateral_vel(env):
    cfg, st, _ = _r(env)
    return core.term_lateral_vel(st, cfg)


def reward_orientation(env):
    cfg, st, _ = _r(env)
    return core.term_orientation(st, cfg)


def reward_yaw(env):
    cfg, st, _ = _r(env)
    return core.term_yaw(st, cfg)


def reward_yaw_rate(env):
    cfg, st, _ = _r(env)
    return core.term_yaw_rate(st, cfg)


def reward_action_rate(env):
    cfg, st, _ = _r(env)
    return core.term_action_rate(st, cfg)


def reward_torques(env):
    cfg, st, _ = _r(env)
    return core.term_torques(st, cfg)


def reward_collision(env):
    cfg, st, _ = _r(env)
    return core.term_collision(st, cfg)


def reward_success(env):
    cfg, st, ctx = _r(env)
    success = core.is_success(st, ctx.cfg, ctx.goal_z)
    return core.term_success(st, cfg, env.termination_manager.terminated, success)


def reward_fall(env):
    cfg, st, ctx = _r(env)
    success = core.is_success(st, ctx.cfg, ctx.goal_z)
    return core.term_fall(st, cfg, env.termination_manager.terminated, success)
