"""奖励函数：每一项一个独立函数，权重在 config.RewardCfg 里同名对应。

约定：
  * 每个 `_xxx(r, cfg)` 返回**该项的原始值**，不带权重；
  * 权重是 `RewardCfg` 里的同名字段（`reward.progress` <-> `RewardCfg.progress`）；
  * `compute_reward()` 负责加权求和，并返回每一项的分项值。

分项值会一路传进 `info` 再记到 TensorBoard，便于分辨是"策略学会了站着不动"还是"某一项写反了号"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RewardCfg, quat_to_rpy


@dataclass
class RewardInput:
    """算一步奖励需要的全部量，由 env 填好。"""

    action: np.ndarray  # 本步动作（12 维）
    prev_action: np.ndarray  # 上一步动作
    q: np.ndarray  # 关节角，执行器顺序
    dq: np.ndarray  # 关节角速度
    tau: np.ndarray  # 关节力矩
    quat: np.ndarray  # 姿态四元数 w x y z
    base_v: np.ndarray  # 世界系线速度 (vx, vy, vz)
    base_w: np.ndarray  # 机体系角速度
    base_pos: np.ndarray  # 世界系位置 (x, y, z)
    prev_x: float  # 上一步的 x（progress 要差分）
    terrain_h: float  # 当前脚下的地形高度
    prev_terrain_h: float  # 上一步的地形高度
    terrain_level: int  # 当前站上第几级台阶（0=还在平地/槛上）
    prev_terrain_level: int  # 上一步的第几级
    cmd_vx: float  # 本回合的前进速度指令
    contact_force: float  # 非脚部件与地形的接触法向力之和
    terminated: bool  # 本步是否因为摔倒/成功而终止
    success: bool  # 本步是否到达终点


# ------------------------------------------------------------------ 前进 / 高度


def _corridor(y: float, cfg: RewardCfg) -> float:
    """走廊权重 w(y)：y=0 处为 1，到 |y|=progress_y_gate 平滑降到 0。必须是连续的。"""
    a = min(abs(y) / max(cfg.progress_y_gate, 1e-6), 1.0)
    return 1.0 - a**3


def _potential_x(x: float, cfg: RewardCfg) -> float:
    """前进势能 Φ(s) = x（只算目标点之前的部分）。走廊权重**不在这里**乘，见 _progress。"""
    return min(x, cfg.progress_x_cap)


def _progress(r: RewardInput, cfg: RewardCfg) -> float:
    """前进奖励：x 势能的**增量**，按走廊权重加权：Δx · w(y)。

    * 用势能增量而不是"每步付一次"的速度年金；
    * 不乘 γ（这里和 _climb 都刻意不乘）；
    * 走廊权重 w(y) 乘在**增量**上，不乘在累积势能上。

    三条都是被实测逼出来的，推导见 AGENT.md §5-B1 / §5-B2 / §5-B3。
    """
    prev = _potential_x(float(r.prev_x), cfg)
    cur = _potential_x(float(r.base_pos[0]), cfg)
    return float((cur - prev) * _corridor(float(r.base_pos[1]), cfg))


def _track_lin_vel(r: RewardInput, cfg: RewardCfg) -> float:
    """速度跟踪，高斯型，最优点就是指令速度。

    尾部减掉 vx=0 处的取值，使得零动作原地不动精确得 0 分（理由见 AGENT.md §5-B1）。
    """
    floor = float(np.exp(-(r.cmd_vx**2) / cfg.track_sigma))
    return float(max(0.0, np.exp(-((r.base_v[0] - r.cmd_vx) ** 2) / cfg.track_sigma) - floor))


def _climb(r: RewardInput, cfg: RewardCfg) -> float:
    """爬升奖励：地形势能的增量 γ·h(x') − h(x)。

    用地形高度增量而不是机身高度增量——后者抬屁股/弹跳就能刷分。权重必须给足
    （推理见 AGENT.md §5-B4）。`climb_gamma` 为 1.0，即不乘 γ（见 AGENT.md §5-B2）。
    """
    return float(cfg.climb_gamma * r.terrain_h - r.prev_terrain_h)


def _level_bonus(r: RewardInput, cfg: RewardCfg) -> float:
    """每登上一级新台阶给一次性的分。

    climb 是势能增量，只在"跨上台阶沿的那一步"出现一次，很容易被动作噪声淹没；
    台阶数是个**离散**信号，直接告诉策略"你上了一个新高度"，比连续势能好学好认。
    只涨不跌：掉下台阶不重复扣（掉下去由 climb 项罚），否则一次失误会被连罚两遍。
    """
    return float(max(0, r.terrain_level - r.prev_terrain_level))


def _base_height(r: RewardInput, cfg: RewardCfg) -> float:
    """机身离当地地形的高度**偏差**（正值，权重是负的 -> 惩罚）。平地和顶平台目标值相同。

    返回**线性**偏差而不是 exp(-err²)，并且带死区 height_dead_zone——
    两者都是为了让"站着不动"不被奖励、让跨台阶的跨坐姿势不挨罚（见 AGENT.md §5-B1 / §5-B4）。
    """
    err = abs(float(r.base_pos[2]) - r.terrain_h - cfg.height_target)
    return float(max(0.0, err - cfg.height_dead_zone))


def _lateral(r: RewardInput, cfg: RewardCfg) -> float:
    """别跑偏到地形外面去。"""
    return float(r.base_pos[1] ** 2)


# ------------------------------------------------------------------ 姿态


def _orientation(r: RewardInput, cfg: RewardCfg) -> float:
    """机身保持水平。"""
    roll, pitch, _ = quat_to_rpy(r.quat)
    return float((1.0 - np.cos(roll)) + (1.0 - np.cos(pitch)))


def _yaw(r: RewardInput, cfg: RewardCfg) -> float:
    """航向对准 +x。地形关于 y 对称、外面又是无限平地，不压住航向就会横着漂出去。"""
    yaw = quat_to_rpy(r.quat)[2]
    return float(1.0 - np.cos(yaw))


def _yaw_rate(r: RewardInput, cfg: RewardCfg) -> float:
    """别原地打转。"""
    return float(r.base_w[2] ** 2)


# ------------------------------------------------------------------ 平滑 / 能耗


def _action_rate(r: RewardInput, cfg: RewardCfg) -> float:
    """动作别抖。"""
    return float(np.sum((r.action - r.prev_action) ** 2))


def _torques(r: RewardInput, cfg: RewardCfg) -> float:
    """省点力矩。"""
    return float(np.sum(r.tau**2))


def _collision(r: RewardInput, cfg: RewardCfg) -> float:
    """非脚部件（大腿、小腿、机身）撞地形的惩罚，按接触力缩放。

    只是惩罚、**不终止**，并且减掉一个底噪 collision_force_free：
    用小腿顶着台阶沿蹭上去是爬不是撞（理由见 AGENT.md §5-B4）。
    """
    return float(max(0.0, r.contact_force - cfg.collision_force_free))


# ------------------------------------------------------------------ 终止


def _success(r: RewardInput, cfg: RewardCfg) -> float:
    return 1.0 if r.success else 0.0


def _fall(r: RewardInput, cfg: RewardCfg) -> float:
    """摔倒/出界终止（成功不算）。"""
    return 1.0 if (r.terminated and not r.success) else 0.0


# 顺序即打印顺序：正向项在前，惩罚项在后，终止项垫底。
TERMS = (
    ("progress", _progress),
    ("track_lin_vel", _track_lin_vel),
    ("climb", _climb),
    ("level_bonus", _level_bonus),
    ("base_height", _base_height),
    ("lateral", _lateral),
    ("orientation", _orientation),
    ("yaw", _yaw),
    ("yaw_rate", _yaw_rate),
    ("action_rate", _action_rate),
    ("torques", _torques),
    ("collision", _collision),
    ("success", _success),
    ("fall", _fall),
)


def compute_reward(r: RewardInput, cfg: RewardCfg) -> tuple[float, dict[str, float]]:
    """加权求和，返回 (总奖励, {项名: 加权后的分项值})。"""
    parts = {name: float(getattr(cfg, name)) * float(fn(r, cfg)) for name, fn in TERMS}
    return float(sum(parts.values())), parts
