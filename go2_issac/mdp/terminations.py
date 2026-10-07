"""终止判据：四个，全部转调 `core.is_*`。

**成功和摔倒都是 `terminated`，超时才是 `truncated`** —— 这一条和 MuJoCo 侧
（`env.py:214-221`）以及 SB3 的 `TimeLimit` 语义完全一致，rsl_rl 靠
`extras["time_outs"]` 对超时做 bootstrap（对应 SB3 的 `TimeLimit.truncated`）。
判据本身一处都没重写：`core.is_success / is_fallen / is_flipped / is_out_of_course`，
`tests/core_parity.py` 用真环境对过边界值（差 1e-9 都测）。

碰撞**不**终止，只扣分（`reward_collision`）。
"""
from __future__ import annotations

from go2_issac import core
from go2_issac.mdp import state


def _term(env) -> dict:
    ctx = state._ctx(env)
    return core.terminations(state.get_state(env), ctx.cfg, ctx.goal_z)


def success(env):
    """站上 0.92 m 的顶平台（还要够高、y 在 ±1 内，见 `EnvCfg.goal_*`）。"""
    return _term(env)["success"]


def fallen(env):
    """机身贴着当地地形（相对高度 < `fall_clearance=0.10`）。"""
    return _term(env)["fallen"]


def flipped(env):
    """|roll| 或 |pitch| > 1.0 rad。"""
    return _term(env)["flipped"]


def out_of_course(env):
    """横向跑出 |y|>2.5，或倒退到 x<-1.0（外面是平地，不拦就能绕过去刷前进奖励）。"""
    return _term(env)["out_of_course"]
