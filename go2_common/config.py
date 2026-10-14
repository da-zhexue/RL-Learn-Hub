"""go2_sac / go2_ppo 共用的常量、环境和奖励配置、姿态小工具。

算法**无关**的那一层：两个训练入口（SAC / PPO）用的是同一个环境、同一套奖励，
所以这些只有一份。算法专属的超参在各自的 `TrainCfg` 里（`go2_sac/config.py` / `go2_ppo/config.py`）。

这里只依赖 numpy，**不 import mujoco**：reward / train / play 都要能在不建仿真环境的情况下
导入配置（play 要读 config.json、reward 要能单独做自检）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ 路径

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = str(REPO_ROOT / "unitree_mujoco" / "unitree_robots" / "go2" / "scene.xml")
MODELS_DIR = REPO_ROOT / "models"
RUNS_DIR = REPO_ROOT / "runs"
CTRL_DIR = REPO_ROOT / "ctrl"

# ------------------------------------------------------------------ 常量

DEFAULT_Q = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)

SIM_DT = 0.005
DECIMATION = 4

KP = 60.0
KD = 3.5

OBS_GYRO_SCALE = 0.25
OBS_CMD_SCALE = 2.0
OBS_DQ_SCALE = 0.05
OBS_CLIP = 10.0
OBS_DIM = 47


def obs_dim(privileged: bool = False) -> int:
    """观测维度。privileged=True 时多 1 维"机身高出当地地形的高度"。"""
    return OBS_DIM + (1 if privileged else 0)


# ------------------------------------------------------------------ 姿态小工具


def quat_to_rpy(quat) -> np.ndarray:
    """四元数 (w, x, y, z) -> 欧拉角 (roll, pitch, yaw)，ZYX 顺序，单位 rad。"""
    w, x, y, z = (float(v) for v in quat)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def projected_gravity(quat) -> np.ndarray:
    """世界系重力方向在机体系下的分量（单位向量）。"""
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [2.0 * (w * y - x * z), -2.0 * (y * z + w * x), 2.0 * (x * x + y * y) - 1.0],
        dtype=np.float64,
    )


# ------------------------------------------------------------------ 配置


@dataclass
class EnvCfg:
    """环境配置。默认值就是"直接能训"的一套，注释里写了实测依据。"""

    scene: str = DEFAULT_SCENE
    terrain: str = "full"  # flat / steps / full，见 terrain.build_model
    terrain_scale: float = 1.0

    # --- 时序与增益
    sim_dt: float = SIM_DT
    decimation: int = DECIMATION
    kp: float = KP
    kd: float = KD
    action_scale: float = 0.6
    joint_margin: float = 0.05  # 目标角离关节硬限位保留的余量，免得一直顶在限位上

    # --- 回合
    max_episode_s: float = 20.0
    cmd_vx: tuple = (0.4, 0.8)  # 每回合随机的前进速度指令，m/s

    # --- 重置分布
    reset_x: tuple = (-0.3, 0.3)
    reset_y: float = 0.15  # y 在 ±0.15 内随机
    # 可选地按比例从近台阶窗口出生；默认 0 保持旧分布与 RNG 消耗不变。
    reset_x_stair: tuple = (1.40, 1.85)
    reset_frac_stair: float = 0.0
    reset_yaw: float = 0.20  # 偏航在 ±0.20 rad 内随机
    reset_z: tuple = (0.30, 0.33)  # home 关键帧的脚底会陷进地面约 18 mm，抬高一点让它自己落稳

    # --- 观测
    privileged: bool = False

    # --- 终止 / 成功判据
    fall_clearance: float = 0.10  # 机身低于当地地形这么多就判摔倒（阈值依据见 AGENT.md §5-C3）
    flip_rad: float = 1.0  # |roll| 或 |pitch| 超过它判翻转（阈值依据见 AGENT.md §5-C3）
    out_y: float = 2.5
    out_x_back: float = -1.0
    goal_x: float = 3.30
    goal_z: float = 1.05
    goal_clearance: float = 0.15  # 站上顶面时 base 约比顶面高 0.27，留 0.15 表示确实站在高处
    goal_y: float = 1.0


@dataclass
class RewardCfg:
    """奖励权重。**项名 = reward.py 里的函数名，权重就是这里的同名字段**，一一对应。
    """

    # --- 前进 / 高度
    progress: float = 100.0
    track_lin_vel: float = 1.0  # 速度跟踪（步态质量，金额很小）
    climb: float = 340.0
    level_bonus: float = 20.0  # 每登上一级新台阶的一次性奖励，离散信号比连续势能好认
    base_height: float = -0.3  # **惩罚**：离当地地形的高度偏差（项函数返回 |err|），站对了是 0
    lateral: float = 0.0
    lateral_vel: float = -5.0

    # --- 姿态
    orientation: float = -0.5
    yaw: float = -1.0  # 地形关于 y 对称、外面是无限平地，不压住航向就会横着漂
    yaw_rate: float = -0.25

    # --- 平滑 / 能耗
    action_rate: float = -0.05
    torques: float = -1e-4
    collision: float = -0.002  # 按接触力缩放：擦碰不等于猛撞

    # --- 终止
    success: float = 450.0
    fall: float = -25.0

    # --- 各项的内部参数
    track_sigma: float = 0.1
    height_target: float = 0.27  # 站立时 base 离地高度
    height_dead_zone: float = 0.05  # 高度偏差的死区（为什么需要见 AGENT.md §5-B4）
    collision_force_free: float = 25.0  # 接触力的免罚底噪，牛顿（见 AGENT.md §5-B4）
    climb_gamma: float = 1.0  # climb 项的折扣，**必须为 1.0**（推导见 AGENT.md §5-B2）
    progress_y_gate: float = 1.25  # 前进奖励的走廊宽度闸门（防刷分，见 AGENT.md §5-B5）
    progress_x_cap: float = 3.5  # 过了这里不再给前进奖励


def to_jsonable(obj) -> dict:
    """把配置数据类转成可以写进 config.json 的普通 dict（tuple 转 list）。"""
    return asdict(obj)
