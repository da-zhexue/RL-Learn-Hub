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
    terrain_h: float
    prev_terrain_h: float  # 上一步的支撑面高度
    climb_h: float  # climb 项用的势能（按 cfg.climb_stat 选的统计量）
    prev_climb_h: float
    terrain_level: int  # 当前站上第几级台阶（0=还在平地/槛上）
    prev_terrain_level: int  # 本步之前本回合到过的最高级（包含出生级数）
    cmd_vx: float  # 本回合的前进速度指令
    contact_force: float  # 非脚部件与地形的接触法向力之和
    terminated: bool  # 本步是否因为摔倒/成功而终止
    success: bool  # 本步是否到达终点
    # 逐脚的量（顺序与 env.foot_geom_ids 一致，4 只脚）
    foot_clearance: np.ndarray  # 脚底离**它正下方**地形的高度，可能为负（脚陷在方块里）
    foot_vx: np.ndarray  # 脚的世界系前向速度，用来判断"在迈步"而不是站着抬脚
    foot_air: np.ndarray  # 脚是否离地（不接触地形）
    clearance_target: float  # 本回合的目标离地间隙（由地形高度算出）
    foot_air_time: np.ndarray  # 本次摆动时长：**只在该脚落地那一步**非零，否则 0
    foot_clearance_ema: np.ndarray  # 每只脚离地高度的低通滤波（拖地脚会明显偏低）


# ------------------------------------------------------------------ 前进 / 高度


def _corridor(y: float, cfg: RewardCfg) -> float:
    """走廊权重 w(y)：y=0 处为 1，到 |y|=progress_y_gate 平滑降到 0。必须是连续的。"""
    a = min(abs(y) / max(cfg.progress_y_gate, 1e-6), 1.0)
    return 1.0 - a**3


def _potential_x(x: float, cfg: RewardCfg) -> float:
    """前进势能 Φ(s) = x（只算目标点之前的部分）。走廊权重**不在这里**乘，见 _progress。"""
    return min(x, cfg.progress_x_cap)


def _progress(r: RewardInput, cfg: RewardCfg) -> float:
    """前进奖励：x 势能的**增量**，按走廊权重加权：Δx · w(y)。"""
    prev = _potential_x(float(r.prev_x), cfg)
    cur = _potential_x(float(r.base_pos[0]), cfg)
    return float((cur - prev) * _corridor(float(r.base_pos[1]), cfg))


def _track_lin_vel(r: RewardInput, cfg: RewardCfg) -> float:
    """速度跟踪，高斯型，最优点就是指令速度。"""
    floor = float(np.exp(-(r.cmd_vx**2) / cfg.track_sigma))
    return float(max(0.0, np.exp(-((r.base_v[0] - r.cmd_vx) ** 2) / cfg.track_sigma) - floor))


def _climb(r: RewardInput, cfg: RewardCfg) -> float:
    """爬升奖励：地形势能的增量 γ·h(x') − h(x)。

    h 是四只脚下方地形高度的统计量（cfg.climb_stat，默认中位数）。
    用中位数时，"两只前脚搭上台阶"这一段完全没有梯度 —— 这正是 0.15 m 台阶爬不上去的
    一个直接原因（见 RewardCfg.climb_stat）。
    """
    return float(cfg.climb_gamma * r.climb_h - r.prev_climb_h)


def _level_bonus(r: RewardInput, cfg: RewardCfg) -> float:
    """每登上一级新台阶给一次性的分。"""
    return float(max(0, r.terrain_level - r.prev_terrain_level))


def _feet_clearance(r: RewardInput, cfg: RewardCfg) -> float:
    """**摆动腿**抬到目标间隙之前给正奖励，用来学会"把脚抬过更高的台阶"。

    只有"离地 + 正在往前迈"的脚才算：站着原地抬脚不给分（否则可以刷分）。
    目标高度由 env 按地形算（= 台阶高度 + 余量），所以它随课程自动缩放。
    """
    if r.clearance_target <= 0.0:
        return 0.0
    gate = np.asarray(r.foot_air, dtype=bool) & (np.asarray(r.foot_vx) > cfg.clearance_vx)
    lift = np.minimum(np.maximum(np.asarray(r.foot_clearance), 0.0), r.clearance_target)
    return float(np.sum(np.where(gate, lift, 0.0)))


def _feet_air_time(r: RewardInput, cfg: RewardCfg) -> float:
    """摆动**时长**奖励：一只脚落地时结算它这次离地了多久（封顶 air_time_target）。

    和 `_feet_clearance` 互补：clearance 管"抬多高"，这一项管"离地多久"。
    两者都只认"真的迈出去"的脚，但关键区别是：这一项**按落地事件结算**，
    一条从不落地的拐杖腿永远拿不到分（实测 2026-10-17：逐脚求和的 clearance
    会被"三条腿跳、RR 当拐杖"刷，见 AGENT.md §5-B9）。
    指令太小时整项归零：站着原地抬脚不该有收益（legged_gym 同款闸门）。
    """
    if abs(float(r.cmd_vx)) < cfg.air_time_cmd_gate:
        return 0.0
    return float(np.sum(np.minimum(np.asarray(r.foot_air_time), cfg.air_time_target)))


def _foot_swing_height(r: RewardInput, cfg: RewardCfg) -> float:
    """**最差那只脚**的"近期平均离地高度"——瓶颈式抬脚塑形，专门反"牺牲一条腿"。

    为什么不能用求和：逐脚求和的 `_feet_clearance` 在"牺牲 RR 当拐杖、三条腿跳"时
    只损失 1/4 的分，实测 PPO 就选了这条路（RR 100% 时间 <0.005 m，另外三只抬到
    0.10~0.15 m，见 AGENT.md §5-B9）。这里取**四只脚的最小值**，最差的那只决定整项，
    必须四条腿都周期性抬起来才拿得到分。

    用 EMA 而不是瞬时 clearance：贴地脚也有 20~30% 的步被接触判断为"离地"，但只有
    1~2 mm——瞬时量会被这种抖动骗过去。alpha=0.995 在 50 Hz 下时间常数约 4 s，
    所以"抬一次然后一直拖着"最多吃几秒分，之后必须再抬。
    """
    if abs(float(r.cmd_vx)) < cfg.air_time_cmd_gate:
        return 0.0
    ema = np.minimum(np.asarray(r.foot_clearance_ema), cfg.clearance_target)
    return float(np.min(ema))


def _base_height(r: RewardInput, cfg: RewardCfg) -> float:
    """机身离当地地形的高度**偏差**（正值，权重是负的 -> 惩罚）。平地和顶平台目标值相同。"""
    err = abs(float(r.base_pos[2]) - r.terrain_h - cfg.height_target)
    return float(max(0.0, err - cfg.height_dead_zone))


def _lateral(r: RewardInput, cfg: RewardCfg) -> float:
    """别跑偏到地形外面去（**权重已置 0**，函数体保留，理由见下面 _lateral_vel 的注释）。"""
    return float(r.base_pos[1] ** 2)


def _lateral_vel(r: RewardInput, cfg: RewardCfg) -> float:
    """横向速度的平方，**世界系**（= 走廊坐标系里的 vy，正是上面 y² 的时间导数）。"""
    return float(r.base_v[1] ** 2)


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
    ("feet_clearance", _feet_clearance),
    ("feet_air_time", _feet_air_time),
    ("foot_swing_height", _foot_swing_height),
    ("base_height", _base_height),
    ("lateral", _lateral),
    ("lateral_vel", _lateral_vel),
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
