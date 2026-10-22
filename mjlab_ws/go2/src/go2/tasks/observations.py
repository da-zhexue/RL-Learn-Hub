from __future__ import annotations

from . import core
from .context import _ctx
from .state import get_state


def policy_obs(env):
    ctx = _ctx(env)
    return core.obs(
        get_state(env),
        ctx.cfg,
        core.default_joint_pos(ctx.cfg, env.scene["robot"].data.joint_pos_limits[0][ctx.joint_ids]),
        privileged=ctx.cfg.privileged,
    )
