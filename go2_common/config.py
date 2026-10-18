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

# 地形高度扫描的默认采样网格（机体系，单位 m）。见 EnvCfg.height_scan。
# 只覆盖身体正前方 0~0.8 m：爬台阶要的是"我前面那级台阶有多高"，不是周围一圈。
SCAN_X = (0.0, 0.2, 0.4, 0.6, 0.8)
SCAN_Y = (-0.2, 0.0, 0.2)


def scan_dim(height_scan: bool, scan_x, scan_y) -> int:
    """高度扫描的维数；height_scan=False 时是 0。"""
    return len(scan_x) * len(scan_y) if height_scan else 0


def obs_dim(privileged: bool = False, n_scan: int = 0) -> int:
    """观测维度 = 47 本体感觉 + 1 特权(可选) + 高度扫描(可选)。

    n_scan 默认 0：go2_issac/core.py 和旧 config.json 都按 47/48 走。
    加扫描是**新增维度**，旧模型不能续训（SB3 会在 load 时报形状不匹配，不会静默）。
    """
    return OBS_DIM + (1 if privileged else 0) + int(n_scan)


# ------------------------------------------------------------------ 姿态小工具


def quat_to_rpy(quat) -> np.ndarray:
    """四元数 (w, x, y, z) -> 欧拉角 (roll, pitch, yaw)，ZYX 顺序，单位 rad。"""
    w, x, y, z = (float(v) for v in quat)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_rotate(quat, vec) -> np.ndarray:
    """用四元数 (w,x,y,z) 旋转向量 vec（可以是 (3,) 或 (N,3)）。

    高度扫描用它把"机体系前方 0.8 m"旋成世界系偏移：狗在台阶上抬头时，
    不旋转的话采样点会指到斜上方去。
    """
    w, x, y, z = (float(v) for v in quat)
    q = np.array([x, y, z], dtype=np.float64)
    arr = np.asarray(vec, dtype=np.float64)
    single = arr.ndim == 1
    u = np.atleast_2d(arr)
    qb = np.broadcast_to(q, u.shape)
    t = 2.0 * np.cross(qb, u)
    out = u + w * t + np.cross(qb, t)
    return out[0] if single else out


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

    # --- 地形感知（height scan）
    # 47 维本体观测里没有任何地形信息：实测盲策略在 0.10~0.11 m 台阶就到顶，
    # 而"前面那级台阶有多高"对策略完全未知。打开后追加
    # len(scan_x)×len(scan_y) 维 = 机体系采样点下方的地形高度 − 机身高度（clip ±1）。
    # **开关改变观测维度，旧模型不能续训**；真机上要订阅 rt/utlidar 的高度图。
    height_scan: bool = False
    scan_x: tuple = SCAN_X
    scan_y: tuple = SCAN_Y

    # --- 终止 / 成功判据
    fall_clearance: float = 0.10  # 机身低于当地地形这么多就判摔倒（阈值依据见 AGENT.md §5-C3）
    flip_rad: float = 1.0  # |roll| 或 |pitch| 超过它判翻转（阈值依据见 AGENT.md §5-C3）
    out_y: float = 2.5
    out_x_back: float = -1.0
    goal_x: float = 3.30
    goal_z: float = 1.05
    goal_clearance: float = 0.15  # 站上顶面时 base 约比顶面高 0.27，留 0.15 表示确实站在高处
    goal_y: float = 1.0

    @property
    def n_scan(self) -> int:
        """高度扫描的维数（height_scan=False 时为 0）。asdict() 不会带上它。"""
        return scan_dim(self.height_scan, self.scan_x, self.scan_y)


@dataclass
class RewardCfg:
    """奖励权重。**项名 = reward.py 里的函数名，权重就是这里的同名字段**，一一对应。
    """

    # --- 前进 / 高度
    progress: float = 100.0
    track_lin_vel: float = 1.0  # 速度跟踪（步态质量，金额很小）
    climb: float = 340.0
    level_bonus: float = 20.0  # 每登上一级新台阶的一次性奖励，离散信号比连续势能好认
    # **惩罚**：离当地地形的高度偏差（项函数返回 |err|），站对了是 0。
    # 10M 步 PPO 平地模型的实测 base 只有 0.190 m（离 target 0.27 差 0.073），但旧值
    # -0.3 × (0.073−0.05 死区) = **-0.007/步**，相对 track_lin_vel 的 +0.93/步 等于没有代价，
    # 于是"低蹲 + 拖脚"成了免费解（肚子最低点 0.087 m，比 0.08 m 的槛只高 7 mm）。
    # 实测 -1.5 热身 1.5M 步只把机身从 0.190 抬到 0.218，且开始往 1.47× 速度的三腿跳漂，
    # 所以再加一档到 -3.0；要更快见效就单独先做一节"站高"课，再打开 feet_clearance。
    base_height: float = -3.0
    lateral: float = 0.0
    lateral_vel: float = -5.0

    # --- 姿态
    orientation: float = -0.5
    yaw: float = -1.0  # 地形关于 y 对称、外面是无限平地，不压住航向就会横着漂
    yaw_rate: float = -0.25

    # --- 迈步（**密集塑形，用来爬上更高的台阶**）
    # 摆动腿抬到 clearance_target 之前的正奖励。没有这一项时，"抬多高"这件事
    # 完全没有梯度：狗在 0.06~0.09 m 的台阶上用小碎步就够了，而 0.15 m 需要
    # 完全不同的步态（前脚要抬到 0.17 才落得上顶面），稀疏的 climb/success
    # 信号在"抬不够高"时给不出任何方向。权重不宜大：抬脚本身不是任务。
    #
    # 2026-10-17 实测（平地 PPO 13M 步）：这一项一直是 0，因为 flat 没有台阶 ->
    # clearance_target=0 -> 函数直接 return 0，而权重本身又是 0 -> 双重关闭。
    # 结果 RR 脚 100% 时间 <0.005 m（纯拖地），抬得最高的 FL 只有 0.062 m。
    # 所以平地也必须开：显式 clearance_target=0.08，权重 3.0（2 只摆动腿抬满
    # 约 +0.4/步，与 track_lin_vel 的 +0.93/步 同一量级，不会盖过速度）。
    feet_clearance: float = 3.0
    # 摆动**时长**奖励（落地时结算本次离地时长，封顶 air_time_target）。
    # 为什么还要这一项：clearance 是逐脚求和的，实测会被"牺牲 RR 当拐杖、三条腿跳"刷分
    # （RR 100% 时间贴地还照样拿其余三条腿的分）。按落地事件结算 + 下面的着地超时罚，
    # 才逼四条腿都真正迈步。量级：trot 每条腿一次落地约 0.25~0.35 s × 6 ≈ 1.5~2.1，
    # 4 条腿 / 一个 0.6 s 周期 -> 约 +0.2/步。
    feet_air_time: float = 6.0
    # **瓶颈式**抬脚塑形：四只脚里"近期平均离地高度"最低的那只决定这一项（见
    # reward._foot_swing_height）。用来堵逐脚求和的漏洞——实测 PPO 会把 RR 当拐杖、
    # 三条腿跳，牺牲一条腿只损失 1/4 的求和分。量级：trot 四脚 EMA 约 0.03~0.05 m
    # -> +0.3~0.5/步；拐杖腿 EMA≈0.001 -> 基本拿不到。
    foot_swing_height: float = 10.0
    # --- 平滑 / 能耗
    action_rate: float = -0.05
    torques: float = -1e-4
    collision: float = -0.002  # 按接触力缩放：擦碰不等于猛撞

    # --- 终止
    success: float = 450.0
    fall: float = -25.0

    # --- 各项的内部参数
    track_sigma: float = 0.1
    height_target: float = 0.29  # 站立时 base 离地高度（diag stance：标称站姿可稳到 0.33）
    height_dead_zone: float = 0.02  # 高度偏差的死区（为什么需要见 AGENT.md §5-B4）
    collision_force_free: float = 25.0  # 接触力的免罚底噪，牛顿（见 AGENT.md §5-B4）
    climb_gamma: float = 1.0  # climb 项的折扣，**必须为 1.0**（推导见 AGENT.md §5-B2）
    progress_y_gate: float = 1.25  # 前进奖励的走廊宽度闸门（防刷分，见 AGENT.md §5-B5）
    progress_x_cap: float = 3.5  # 过了这里不再给前进奖励
    # climb 项用四只脚下方地形高度的哪个统计量当势能：
    #   "median" = 老口径（四只脚的中位数）。踩上去两只脚时中位数还在地面 -> 台阶白爬，
    #              要三只脚上去才给分。对 0.15 m 这种高台阶，这一段就是"没有梯度"。
    #   "mean"   = 四只脚的平均。两只前脚搭上台阶就给一半的分，credit 分配平滑得多。
    climb_stat: str = "median"
    # 目标离地间隙 (m)。**显式给一个平地也能用的值**：0 表示"按地形最高一级台阶自动算"，
    # 而 flat/steps 之外只有 full 有台阶，平地训出的就是 target=0 = 没有抬脚梯度。
    # 0.08 = steps 的槛高；台阶地形取 max(它, 台阶高+margin)，见 env.Go2TerrainEnv.__init__。
    air_time_target: float = 0.35  # 摆动时长奖励的封顶（再久不加分，防止刻意拖长）
    air_time_cmd_gate: float = 0.10  # |cmd_vx| 小于它时 feet_air_time 整项归零
    clearance_ema_alpha: float = 0.995  # 逐脚 clearance 的 EMA 系数；50 Hz 下时间常数约 4 s
    clearance_target: float = 0.08
    clearance_margin: float = 0.02  # 自动目标 = 台阶高度 + 它
    clearance_vx: float = 0.02  # 脚的前向速度超过它才算"在迈步"，站着抬脚不给分


def to_jsonable(obj) -> dict:
    """把配置数据类转成可以写进 config.json 的普通 dict（tuple 转 list）。"""
    return asdict(obj)
