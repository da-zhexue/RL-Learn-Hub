from __future__ import annotations

from . import core
from .context import _ctx
from .state import get_state


def success(env):
    ctx = _ctx(env)
    return core.is_success(get_state(env), ctx.cfg, ctx.goal_z)


def fallen(env):
    ctx = _ctx(env)
    return core.is_fallen(get_state(env), ctx.cfg)


def flipped(env):
    ctx = _ctx(env)
    return core.is_flipped(get_state(env), ctx.cfg)


def out_of_course(env):
    ctx = _ctx(env)
    return core.is_out_of_course(get_state(env), ctx.cfg)
