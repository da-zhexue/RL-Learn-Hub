"""14 项奖励，一项一个函数，全部转调 `core.term_*`。

**这里不写任何公式**：每项一行 `return core.term_xxx(...)`。Isaac 的奖励管理器做的是
"权重 × 函数值，再求和"，`go2_common/reward.py` 的 `compute_reward` 做的也是同一件事，
所以把这 14 个函数挂成 14 个 `RewardTermCfg` 之后，两边的分项天然一一对应：
`Episode_Reward/<项名>` 和 TensorBoard 里 MuJoCo 侧记的是同一套名字、同一套权重。
权重值写在 `env_cfg.py` 的 `RewardTermCfg.weight` 上，来源是 `RewardCfg` 的同名字段
（`_weight()` 去读，不抄数字）。

**函数返回的是"没乘权重的原始值"**（可能为负，比如 `base_height` 返回的是偏差量），
乘权重是 Isaac 各版本都做的、也是 `compute_reward` 做的。

`core.TERMS` 是项名和顺序的唯一真源，模块末尾会断言这 14 个函数和它对得上——
漏一项、名字写错，导入时就炸，而不是等到训练时发现奖励少了一块。
"""
from __future__ import annotations

from go2_issac import core
from go2_issac.mdp import state


def _weight(env, name: str) -> float:
    """这一项的权重（来自 `RewardCfg`）。`env_cfg.py` 建 term 时用它，避免两处数字。"""
    return float(getattr(state._ctx(env).reward_cfg, name))


def _r(env):
    """每项都要的两样东西：配置 + 当拍状态。"""
    ctx = state._ctx(env)
    return ctx.reward_cfg, state.get_state(env)


# ------------------------------------------------------------------ 前进 / 高度


def reward_progress(env):
    """前进：x 势能的增量 × 走廊权重（唯一的大额收入来源）。"""
    cfg, st = _r(env)
    return core.term_progress(st, cfg)


def reward_track_lin_vel(env):
    """速度跟踪（高斯），站着不动精确得 0。"""
    cfg, st = _r(env)
    return core.term_track_lin_vel(st, cfg)


def reward_climb(env):
    """爬升：地形势能增量（用地形高度，不用机身高度）。"""
    cfg, st = _r(env)
    return core.term_climb(st, cfg)


def reward_level_bonus(env):
    """每登上一级新台阶的一次性奖励。"""
    cfg, st = _r(env)
    return core.term_level_bonus(st, cfg)


def reward_base_height(env):
    """离当地地形的高度偏差（惩罚项，带死区）。"""
    cfg, st = _r(env)
    return core.term_base_height(st, cfg)


def reward_lateral(env):
    """保持在 y=0 走廊里（惩罚项）。"""
    cfg, st = _r(env)
    return core.term_lateral(st, cfg)


# ------------------------------------------------------------------ 姿态


def reward_orientation(env):
    cfg, st = _r(env)
    return core.term_orientation(st, cfg)


def reward_yaw(env):
    cfg, st = _r(env)
    return core.term_yaw(st, cfg)


def reward_yaw_rate(env):
    cfg, st = _r(env)
    return core.term_yaw_rate(st, cfg)


# ------------------------------------------------------------------ 平滑 / 能耗


def reward_action_rate(env):
    """相邻两步动作差的平方和。

    **用 `st.prev_action`（a_{t-1}），不是 `st.action`（a_t）**——这两个字段值不同，
    写混了这项恒等于 0（自己减自己），而且不会报错。`tests/core_parity.py` 专门盯这个。
    """
    cfg, st = _r(env)
    return core.term_action_rate(st, cfg)


def reward_torques(env):
    cfg, st = _r(env)
    return core.term_torques(st, cfg)


def reward_collision(env):
    """非脚部件撞地形，按接触力缩放（减掉 25 N 的底噪）。**不终止**。"""
    cfg, st = _r(env)
    return core.term_collision(st, cfg)


# ------------------------------------------------------------------ 终止类


def _terminated(env):
    """成功/摔倒这两个布尔量，一次算出来给两项用。"""
    ctx = state._ctx(env)
    st = state.get_state(env)
    return st, ctx, core.terminations(st, ctx.cfg, ctx.goal_z)


def reward_success(env):
    """到达顶平台（站上 0.92 m 那块，见 `EnvCfg.goal_*`）。"""
    st, ctx, term = _terminated(env)
    return core.term_success(st, ctx.reward_cfg, term["any"], term["success"])


def reward_fall(env):
    """摔倒/出界终止（成功不算）。"""
    st, ctx, term = _terminated(env)
    return core.term_fall(st, ctx.reward_cfg, term["any"], term["success"])


# ------------------------------------------------------------------ 一致性自检

#: 项名 -> 函数。`env_cfg.py` 按这个表建 `RewardTermCfg`，顺序即声明顺序。
TERMS: dict[str, object] = {
    "progress": reward_progress,
    "track_lin_vel": reward_track_lin_vel,
    "climb": reward_climb,
    "level_bonus": reward_level_bonus,
    "base_height": reward_base_height,
    "lateral": reward_lateral,
    "orientation": reward_orientation,
    "yaw": reward_yaw,
    "yaw_rate": reward_yaw_rate,
    "action_rate": reward_action_rate,
    "torques": reward_torques,
    "collision": reward_collision,
    "success": reward_success,
    "fall": reward_fall,
}

# 名字/顺序必须和 `go2_common/reward.py` 的 TERMS 一模一样，否则 TensorBoard 上两边对不上
assert tuple(TERMS) == core.TERM_NAMES, (
    f"奖励项名对不上：模块里是 {tuple(TERMS)}，core 里是 {core.TERM_NAMES}")
