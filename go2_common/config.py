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

# 默认站姿，顺序与执行器/DDS 报文一致：FR, FL, RR, RL，每条腿 [hip, thigh, calf]。
# 取自 go2.xml 的 home 关键帧 qpos="0 0 0.27 1 0 0 0 0 0.9 -1.8 ..."。
DEFAULT_Q = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)

# 时序。0.005 s 刻意等于 unitree_mujoco/simulate_python/config.py 里的 SIMULATE_DT，
# 让训练时的积分步长和 DDS 仿真完全一致，缩小 sim-to-sim 差距；
# 每个策略步走 4 个物理步 -> 策略 50 Hz（爬台阶需要比 50 Hz 更快的话再调 decimation）。
SIM_DT = 0.005
DECIMATION = 4

# 关节 PD 增益。这两个值在 DDS(200 Hz) 和直接步进(500 Hz) 两条路径上都实测稳定：
# 站立 z≈0.268 m、|dq|≈0，离 home 约 0.11 rad。
KP = 60.0
KD = 3.5

# 观测缩放/裁剪。观测维度 = 3(角速度) + 3(投影重力) + 3(指令) + 12(关节角) + 12(关节速) + 12(上一步动作)
OBS_GYRO_SCALE = 0.25
OBS_CMD_SCALE = 2.0
OBS_DQ_SCALE = 0.05
OBS_CLIP = 10.0
OBS_DIM = 45


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
    """世界系重力方向在机体系下的分量（单位向量）。

    注意用的是四元数而不是加速度计：DDS 的 LowState 里有四元数，仿真和真机都能算，
    加速度计在动起来以后噪声大得多。
    """
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
    # 地形高度的整体缩放，用来做课程：0.3 时两个槛只有 2.4 cm、六级台阶每级只升 4.5 cm，
    # 1.0 就是原场景。只缩放几何体的高度（size[2]/pos[2]），x/y 脚印不动，所以台阶的踏面深度
    # 不变、只是变矮——这正是想要的"变简单"。0 就是纯平地。
    # **为什么必须做课程**：实测平地训出来的策略在槛高 0.04~0.06 时能爬上去，到 0.08 就突然
    # 0/3 全卡住（断层很硬）。直接从平地跳到 0.08，策略只会停在槛前拿 track_lin_vel 的保底分。
    terrain_scale: float = 1.0

    # --- 时序与增益
    sim_dt: float = SIM_DT
    decimation: int = DECIMATION
    kp: float = KP
    kd: float = KD
    # 关节目标 = 默认站姿 + action_scale * 动作。**这个值决定了狗能跨多高的槛，不是随便调的**。
    #
    # 测法是直接跑：拿手调开环小跑（无反馈）撞槛，扫出「绝对摆幅 -> 能跨过的最高槛」：
    #
    #     action_scale   0.04m  0.06m  0.08m  0.10m  0.12m
    #         0.40        过     卡     卡     卡     卡
    #         0.50        过     过     卡     卡     卡
    #         0.60        过     过     卡     卡     卡
    #         0.80        过     过     过     过     卡
    #
    # 带反馈的策略比这套开环小跑大约好一档（0.40 时开环只能过 0.04，平地训出来的策略能过 0.06）。
    # 场景里那两个槛是 0.08 m，所以要 0.6 才有戏；0.4 时策略会走到 x≈0.94（机身前缘正好顶在
    # x=1.1 的槛面上）就停住，20 万步毫无进展——这不是奖励或探索的问题，是幅度不够。
    action_scale: float = 0.6
    joint_margin: float = 0.05  # 目标角离关节硬限位保留的余量，免得一直顶在限位上

    # --- 回合
    max_episode_s: float = 20.0
    cmd_vx: tuple = (0.4, 0.8)  # 每回合随机的前进速度指令，m/s

    # --- 重置分布
    reset_x: tuple = (-0.3, 0.3)
    reset_y: float = 0.15  # y 在 ±0.15 内随机
    reset_yaw: float = 0.20  # 偏航在 ±0.20 rad 内随机
    reset_z: tuple = (0.30, 0.33)  # home 关键帧的脚底会陷进地面约 18 mm，抬高一点让它自己落稳

    # --- 观测
    # True 时额外给 1 维"机身高出当地地形的高度"。这一维对爬台阶很有用，但 DDS 的
    # LowState 里没有（只有 rt/sportmodestate 才有位置），所以默认关掉，保证策略能直接上真机。
    privileged: bool = False

    # --- 终止 / 成功判据
    fall_clearance: float = 0.10  # 机身低于当地地形这么多就判摔倒（阈值依据见 AGENT.md §5-C3）
    flip_rad: float = 1.0  # |roll| 或 |pitch| 超过它判翻转（阈值依据见 AGENT.md §5-C3）
    out_y: float = 2.5
    out_x_back: float = -1.0
    # 顶台阶顶面 0.92 m，x∈[3.14, 3.66]，平台深 0.52 m。不取 x>=3.6：那里离 0.92 m 悬崖
    # 只剩 6 cm，会奖励"冲过终点再摔下去"。
    goal_x: float = 3.30
    # goal_z 是 env 建好地形后按 terrain.top(goal_x) + goal_clearance 算出来的，**不是**这里的值
    # （为什么不能写死见 AGENT.md §5-C4）。
    goal_z: float = 1.05
    goal_clearance: float = 0.15  # 站上顶面时 base 约比顶面高 0.27，留 0.15 表示确实站在高处
    goal_y: float = 1.0


@dataclass
class RewardCfg:
    """奖励权重。**项名 = reward.py 里的函数名，权重就是这里的同名字段**，一一对应。

    调权重时优先动这里，别去改 reward.py 的函数体。
    """

    # --- 前进 / 高度
    # progress 是这套奖励里唯一的大额收入来源，要压得住其它所有项，而站着不动必须**恒等于 0**。
    # 它现在是**势能** γ·x'−x（见 reward._progress），不再是"每步付一次"的速度年金：
    # 单位从"分/步"变成"分/米"，100.0 对应走起来约 +1~2 分/步、走完全程约 190 分。
    progress: float = 100.0
    track_lin_vel: float = 1.0  # 速度跟踪（步态质量，金额很小）
    # 带折扣的地形势能 γ·h(x')−h(x)。**必须给足量**：0.08 m 的槛 × 2.0 只有 0.16 分，
    # 而卡在槛上时 height+collision 一步就 -0.13，爬上去净亏 → 策略学会不爬（实测确凿）。
    #
    # 20.0 是量出来的错值：整段楼梯只涨 0.57 m，20.0 只兑现 11 分，而同期 progress 年金
    # 付了 250 分——爬楼的收益比不爬只高 23%，梯度一直在教策略"别爬"。progress 改成势能后
    # 前进的总收入是 (x_cap−x_0)·100 ≈ 190 分，高度按"值前进的一半"配：0.57·W ≈ 95 → W≈170，
    # 取 100（宁可让高度略轻于水平，也不要它盖过主导项把步态压垮）。
    climb: float = 100.0
    level_bonus: float = 5.0  # 每登上一级新台阶的一次性奖励，离散信号比连续势能好认
    base_height: float = -0.3  # **惩罚**：离当地地形的高度偏差（项函数返回 |err|），站对了是 0
    lateral: float = -0.3  # 保持在 y=0 走廊里

    # --- 姿态
    orientation: float = -0.5
    yaw: float = -1.0  # 地形关于 y 对称、外面是无限平地，不压住航向就会横着漂
    yaw_rate: float = -0.25

    # --- 平滑 / 能耗
    action_rate: float = -0.05
    torques: float = -1e-4
    collision: float = -0.002  # 按接触力缩放：擦碰不等于猛撞

    # --- 终止
    success: float = 20.0
    # 摔倒的代价。progress 改成势能后，"多活一会儿"不再自动等于"多拿分"——但在楼梯中途摔下去
    # 仍然是净亏：climb 项会把已经兑现的 0.57 m 高度收回去，而重爬要再顶一遍碰撞和力矩惩罚。
    # -5.0 那次是配着"progress 年金"估的，那时摔跤的主要代价是丢掉剩余年金（几百步×1分），
    # 年金没了，这个数就得自己把它补上，否则策略会拿"摔了算逑"去赌快跑。
    fall: float = -25.0

    # --- 各项的内部参数
    # 高斯速度跟踪的分母（形式上是 σ²）。0.25 时站着不动还能拿 exp(-0.6²/0.25)=0.20，
    # 是"站着不动"第二大的一笔白拿收入；收到 0.1 后只剩 0.027。
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
