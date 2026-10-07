"""观测 / 奖励 / 终止判据 / 命令：Isaac 侧的唯一实现，纯 torch，不 import isaaclab。

逐条对照 `go2_common/env.py` 的 `_obs()` / `step()` 和 `go2_common/reward.py` 的 14 项。
奖励**权重**（`RewardCfg`）、观测缩放（`OBS_*`）、`action_scale`、`kp/kd`、
成功/摔倒判据的阈值都从 `go2_common` 读，这里**不复制**任何常量——
所以 go2_ppo 那边调完权重，Isaac 这边跟着变，不会出现两份配置漂移。

靠 `tests/core_parity.py` 和真 MuJoCo 环境逐位对拍（本机无卡就能跑）。

关于精度：Isaac Lab 的状态张量是 float32，这里的采样/比较全部用 float64 算
（盒子表也是 float64），只在输出观测时按 `env.py` 的做法转 float32。
这样"同一个输入 -> 同一个输出"和 MuJoCo 侧一致；唯一的差别是输入本身带 float32 舍入
（约 1e-8 m），它只可能在采样点**恰好落在方块棱上**时让当地高度差一个台阶
（概率 ~1e-8），对训练无影响，但别指望和 MuJoCo 逐位相等——`tests/core_parity.py`
用的是 float64 输入，量的才是纯粹的语义一致性。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from go2_common.config import (
    DEFAULT_Q,
    OBS_CLIP,
    OBS_CMD_SCALE,
    OBS_DQ_SCALE,
    OBS_GYRO_SCALE,
    RewardCfg,
    obs_dim,
)
# 项名与顺序的唯一真源：直接用 reward.py 的 TERMS，不重抄一遍名字
from go2_common.reward import TERMS as _MJ_TERMS
from go2_issac import course

TERM_NAMES = tuple(name for name, _fn in _MJ_TERMS)

F32 = torch.float32
F64 = torch.float64


# ------------------------------------------------------------------ 姿态小工具


def projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    """世界系重力方向在机体系下的分量（单位向量），与 `config.projected_gravity` 同式。

    对齐姿态（w=1）时是 (0, 0, -1)。**这一维的符号最容易搞错**，`smoke.py --sanity`
    会在站立姿态上专门断言一次。
    """
    w, x, y, z = quat.unbind(-1)
    return torch.stack(
        [2.0 * (w * y - x * z), -2.0 * (y * z + w * x), 2.0 * (x * x + y * y) - 1.0], dim=-1
    )


def quat_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """四元数 (w, x, y, z) -> (roll, pitch, yaw)，ZYX，与 `config.quat_to_rpy` 同式。"""
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


# ------------------------------------------------------------------ 地形（torch 版）


class CourseTerrain:
    """`course.CourseHeight` 的 torch 版：盒子表来自 `course.py`（唯一真源）。

    语义与 `terrain.TerrainHeight` 一致：`top` 取盖住该点的最高方块（没盖住就是 0），
    `level` 数"顶面不高于当地高度的台阶数"。默认 float64：这几步对比在 float32 下会因为
    0.24 m 的采样偏移丢精度，而当地高度差一个台阶就是 0.15 m，不值得省这点。
    """

    def __init__(self, variant: str = "full", scale: float = 1.0,
                 device=None, dtype: torch.dtype = F64):
        ch = course.CourseHeight(variant, scale)
        self.variant, self.scale = variant, scale
        self.device, self.dtype = device, dtype
        self.stair_tops = ch.stair_tops
        # (M, 5) = x_lo, x_hi, y_lo, y_hi, top
        self.boxes = torch.tensor(ch.boxes if ch.boxes else np.zeros((0, 5)),
                                  dtype=dtype, device=device)

    def to(self, device) -> "CourseTerrain":
        out = CourseTerrain.__new__(CourseTerrain)
        out.__dict__.update(self.__dict__)
        out.device = device
        out.boxes = self.boxes.to(device)
        return out

    # ------------------------------------------------------------ 查询

    def top(self, x, y=0.0):
        """当地地形高度。标量进标量出、数组进数组出（与 `course.CourseHeight.top` 对称）。"""
        x, y, scalar = self._as_tensors(x, y)
        if self.boxes.shape[0] == 0:
            out = torch.zeros(torch.broadcast_shapes(x.shape, y.shape),
                              dtype=self.dtype, device=x.device)
        else:
            x_lo, x_hi, y_lo, y_hi, tops = self.boxes.unbind(-1)
            # 和 terrain.py 一样是**闭区间**：`x_lo <= x <= x_hi`
            hit = ((x.unsqueeze(-1) >= x_lo) & (x.unsqueeze(-1) <= x_hi)
                   & (y.unsqueeze(-1) >= y_lo) & (y.unsqueeze(-1) <= y_hi))
            # 集合 {被盖住的方块顶面} ∪ {0} 取最大 == terrain.py 的 h 累加最大值
            out = torch.where(hit, tops, torch.zeros_like(tops)).amax(dim=-1)
        return float(out) if scalar else out

    def _as_tensors(self, x, y) -> tuple[torch.Tensor, torch.Tensor, bool]:
        scalar = not torch.is_tensor(x) and not torch.is_tensor(y) \
            and np.ndim(x) == 0 and np.ndim(y) == 0
        x = torch.as_tensor(x, dtype=self.dtype, device=self.boxes.device)
        y = torch.as_tensor(y, dtype=self.dtype, device=self.boxes.device)
        return x, y, scalar

    def top_footprint(self, x, y=0.0, half_x: float = 0.24, half_y: float = 0.12):
        """身体脚印范围内的最大地形高度。采样点与 `terrain.TerrainHeight` 逐位一致。

        9×3 个采样点，公式刻意写成 `start + delta*i` 并把最后一个点钉成 `stop`：
        `np.linspace` 就是这么算的（`y = delta*arange(n) + start; y[-1] = stop`），
        这样两个版本的采样点**位相等**，不会出现"某个采样点恰好落在方块棱上、
        两边一个算进去一个没算进去"的假阳性。加法可交换，所以 `(x-half_x) + delta*i`
        和 numpy 的 `delta*i + start` 位相等。
        """
        x, y, scalar = self._as_tensors(x, y)
        delta = (2.0 * half_x) / 8.0
        i = torch.arange(9, dtype=self.dtype, device=x.device)
        xs = (x - half_x).unsqueeze(-1) + delta * i
        xs = torch.cat([xs[..., :-1], (x + half_x).unsqueeze(-1)], dim=-1)
        ys = torch.stack([y - half_y, y, y + half_y], dim=-1)
        out = self.top(xs.unsqueeze(-1), ys.unsqueeze(-2)).amax(dim=(-2, -1))
        return float(out) if scalar else out

    def level(self, x, y=0.0):
        """站上第几级台阶（0 = 还在平地/槛上）。"""
        x, y, scalar = self._as_tensors(x, y)
        h = self.top(x, y)
        if not self.stair_tops:
            out = torch.zeros_like(h, dtype=torch.long)
        else:
            tops = torch.tensor(self.stair_tops, dtype=self.dtype, device=h.device)
            out = (tops <= h.unsqueeze(-1) + 1e-6).sum(dim=-1)
        return int(out) if scalar else out


def goal_z_for(cfg, ch: CourseTerrain | course.CourseHeight) -> float:
    """成功判据里的"顶平台高度"= 顶平台处地形高度 + goal_clearance。

    没有台阶的课程（flat / steps）就没有"顶平台"这回事，高度设成够不着（inf），
    和 `env.py:55-59` 同款——否则在平地往前走也能白拿 success（见 AGENT.md §5-C4）。
    """
    return ch.top(cfg.goal_x, 0.0) + cfg.goal_clearance if ch.stair_tops else math.inf


# ------------------------------------------------------------------ 状态


@dataclass
class State:
    """算一步观测/奖励需要的全部量：当前状态 + 上一拍的累积量 + 地形/命令。

    字段顺序和含义对着 `env.py` 的 `RewardInput` 与 `_obs()`：
    线速度是**世界系**、角速度是**机体系**（自由关节的 qvel 就是这个约定）。
    """

    # --- 当前状态，全部 (N, ...)，float64
    base_pos: torch.Tensor      # (N,3) 世界系位置
    base_quat: torch.Tensor     # (N,4) wxyz
    base_lin_vel: torch.Tensor  # (N,3) 世界系线速度
    base_ang_vel: torch.Tensor  # (N,3) 机体系角速度（陀螺）
    q: torch.Tensor             # (N,12) 关节角，执行器/SDK 顺序
    dq: torch.Tensor            # (N,12)
    tau: torch.Tensor           # (N,12) 实际力矩
    contact_force: torch.Tensor  # (N,) 非脚部件与地形的接触法向力之和
    action: torch.Tensor        # (N,12) 本步动作（clip 到 ±1 之后），action_rate 要用

    # --- 地形（由位置算出来，见 CourseTerrain）
    terrain_h: torch.Tensor     # (N,) 当地地形高度（中心点）
    terrain_level: torch.Tensor  # (N,) 站上第几级

    # --- 上一拍
    prev_x: torch.Tensor        # (N,)
    prev_action: torch.Tensor   # (N,12)
    prev_terrain_h: torch.Tensor
    prev_terrain_level: torch.Tensor

    # --- 命令
    cmd_vx: torch.Tensor        # (N,) 本回合的前进速度指令（整回合不变）

    @property
    def n(self) -> int:
        return int(self.base_pos.shape[0])

    def rpy(self) -> torch.Tensor:
        return quat_to_rpy(self.base_quat)


def advance(st: State) -> dict:
    """算出这一步该留给下一拍的累积量。

    顺序不能反（对照 `env.py:245-248`）：**先用旧值算奖励，再推进**。
    """
    return {
        "prev_action": st.action,
        "prev_x": st.base_pos[:, 0],
        "prev_terrain_h": st.terrain_h,
        "prev_terrain_level": st.terrain_level,
    }


# ------------------------------------------------------------------ 观测


def default_joint_pos(cfg, joint_limits: torch.Tensor) -> torch.Tensor:
    """`DEFAULT_Q` 按关节限位夹一下，和 `env.py` 的 `q_default` 同款。

    `joint_limits` 是 `(12, 2)` 的 (lo, hi)，顺序 = 执行器/SDK 顺序。
    Isaac 侧要保证 `soft_joint_pos_limit_factor=1.0`，这样拿到的是硬限位；
    `joint_margin` 那 0.05 是 MuJoCo 侧自己留的余量，这里照抄，免得两边默认站姿不同。
    """
    limits = joint_limits.to(F64)
    q_lo = limits[:, 0] + cfg.joint_margin
    q_hi = limits[:, 1] - cfg.joint_margin
    q = torch.tensor(DEFAULT_Q, dtype=F64, device=limits.device)
    return torch.clamp(q, min=q_lo, max=q_hi)


def obs(st: State, cfg, q_default, privileged: bool = False) -> torch.Tensor:
    """45 维观测（privileged 时 46 维），float32、clip ±10，逐项对照 `env.py:_obs()`。

    组成： 角速度×0.25 | 投影重力 | 指令×2 | q-q_default | dq×0.05 | 上一步动作

    `q_default` 可以是张量也可以是 numpy 数组（对拍时直接喂 `env.q_default`）。

    **最后那 12 维是 `st.action`（本步动作 a_t），不是 `st.prev_action`（a_{t-1}）。**
    `env.py` 的 `step()` 先把 `prev_action` 推进再构造返回的观测，所以步 t 返回的观测里
    装的是 a_t；奖励里的 `action_rate` 要的才是 a_{t-1}。这两个值在同一个 `State` 里
    是两个字段，写混了观测就整体偏一位（对拍能抓到）。
    """
    n = st.n
    q_default = torch.as_tensor(q_default, dtype=F64, device=st.q.device)
    parts = [
        st.base_ang_vel.to(F64) * OBS_GYRO_SCALE,
        projected_gravity(st.base_quat.to(F64)),
        torch.tensor([1.0, 0.0, 0.0], dtype=F64, device=st.base_pos.device)
        * st.cmd_vx.to(F64).unsqueeze(-1) * OBS_CMD_SCALE,
        st.q.to(F64) - q_default.to(F64),
        st.dq.to(F64) * OBS_DQ_SCALE,
        st.action.to(F64),
    ]
    out = torch.cat(parts, dim=-1)
    if privileged:
        # 唯一在 DDS 的 LowState 里拿不到的一维（要用 rt/sportmodestate），默认关
        out = torch.cat([out, (st.base_pos[:, 2] - st.terrain_h).unsqueeze(-1)], dim=-1)
    assert out.shape == (n, obs_dim(privileged)), f"观测维度 {out.shape} 不对"
    return torch.clamp(out, -OBS_CLIP, OBS_CLIP).to(F32)


# ------------------------------------------------------------------ 奖励（14 项）
#
# 每项一个函数，返回**不带权重**的原始值；权重在 RewardCfg 里同名对应，由 compute_reward 求和。
# 逐项对照 reward.py，包括那些看起来多余的 max(0, ...) —— 它们都是被实测逼出来的（见 README 奖励一节）。


def _corridor(y: torch.Tensor, cfg: RewardCfg) -> torch.Tensor:
    a = torch.clamp(torch.abs(y) / max(cfg.progress_y_gate, 1e-6), max=1.0)
    return 1.0 - a**3


def _potential_x(x: torch.Tensor, cfg: RewardCfg) -> torch.Tensor:
    return torch.clamp(x, max=cfg.progress_x_cap)


def term_progress(st: State, cfg: RewardCfg) -> torch.Tensor:
    """前进奖励：x 势能的增量 × 走廊权重。不乘 γ（见 AGENT.md §5-B2）。"""
    prev = _potential_x(st.prev_x, cfg)
    cur = _potential_x(st.base_pos[:, 0], cfg)
    return (cur - prev) * _corridor(st.base_pos[:, 1], cfg)


def term_track_lin_vel(st: State, cfg: RewardCfg) -> torch.Tensor:
    """速度跟踪（高斯），尾部减掉 vx=0 的取值，站着不动精确得 0。"""
    floor = torch.exp(-(st.cmd_vx**2) / cfg.track_sigma)
    return torch.clamp(
        torch.exp(-((st.base_lin_vel[:, 0] - st.cmd_vx) ** 2) / cfg.track_sigma) - floor,
        min=0.0,
    )


def term_climb(st: State, cfg: RewardCfg) -> torch.Tensor:
    """爬升奖励：地形势能增量。用地形高度而不是机身高度（抬屁股就能刷后者）。"""
    return cfg.climb_gamma * st.terrain_h - st.prev_terrain_h


def term_level_bonus(st: State, cfg: RewardCfg) -> torch.Tensor:
    """每登上一级新台阶的一次性奖励。只涨不跌（掉下来由 climb 罚）。"""
    return torch.clamp(st.terrain_level - st.prev_terrain_level, min=0).to(F64)


def term_base_height(st: State, cfg: RewardCfg) -> torch.Tensor:
    """离当地地形的高度偏差（线性 + 死区，不用 exp(-err²)：否则站对了白拿分）。"""
    err = torch.abs(st.base_pos[:, 2] - st.terrain_h - cfg.height_target)
    return torch.clamp(err - cfg.height_dead_zone, min=0.0)


def term_lateral(st: State, cfg: RewardCfg) -> torch.Tensor:
    return st.base_pos[:, 1] ** 2


def term_orientation(st: State, cfg: RewardCfg) -> torch.Tensor:
    rpy = st.rpy()
    return (1.0 - torch.cos(rpy[:, 0])) + (1.0 - torch.cos(rpy[:, 1]))


def term_yaw(st: State, cfg: RewardCfg) -> torch.Tensor:
    return 1.0 - torch.cos(st.rpy()[:, 2])


def term_yaw_rate(st: State, cfg: RewardCfg) -> torch.Tensor:
    return st.base_ang_vel[:, 2] ** 2


def term_action_rate(st: State, cfg: RewardCfg) -> torch.Tensor:
    return ((st.action - st.prev_action) ** 2).sum(dim=-1)


def term_torques(st: State, cfg: RewardCfg) -> torch.Tensor:
    return (st.tau**2).sum(dim=-1)


def term_collision(st: State, cfg: RewardCfg) -> torch.Tensor:
    """非脚部件撞地形，按接触力缩放，减掉底噪（蹭着台阶沿爬上去不算撞）。不终止。"""
    return torch.clamp(st.contact_force - cfg.collision_force_free, min=0.0)


def term_success(st: State, cfg: RewardCfg, terminated: torch.Tensor,
                 success: torch.Tensor) -> torch.Tensor:
    return success.to(F64)


def term_fall(st: State, cfg: RewardCfg, terminated: torch.Tensor,
              success: torch.Tensor) -> torch.Tensor:
    """摔倒/出界终止（成功不算）。"""
    return (terminated & ~success).to(F64)


TERMS = (
    ("progress", term_progress),
    ("track_lin_vel", term_track_lin_vel),
    ("climb", term_climb),
    ("level_bonus", term_level_bonus),
    ("base_height", term_base_height),
    ("lateral", term_lateral),
    ("orientation", term_orientation),
    ("yaw", term_yaw),
    ("yaw_rate", term_yaw_rate),
    ("action_rate", term_action_rate),
    ("torques", term_torques),
    ("collision", term_collision),
    ("success", term_success),
    ("fall", term_fall),
)


def compute_reward(st: State, cfg: RewardCfg, terminated: torch.Tensor,
                   success: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """加权求和，返回 `(总奖励 (N,), {项名: 加权后的分项 (N,)})`。

    和 `reward.compute_reward` 一一对应，分项值同样会记到 TensorBoard，
    便于分辨"策略学会了站着不动"还是"某一项写反了号"。
    """
    parts: dict[str, torch.Tensor] = {}
    for name, fn in TERMS:
        if name in ("success", "fall"):
            raw = fn(st, cfg, terminated, success)
        else:
            raw = fn(st, cfg)
        parts[name] = float(getattr(cfg, name)) * raw
    total = sum(parts.values())
    return total, parts


# ------------------------------------------------------------------ 终止判据


def is_success(st: State, cfg, goal_z: float) -> torch.Tensor:
    """到达顶平台。高度条件不能省：否则在平地往前走同样能白拿这个奖励。"""
    return ((st.base_pos[:, 0] >= cfg.goal_x)
            & (st.base_pos[:, 2] >= goal_z)
            & (torch.abs(st.base_pos[:, 1]) <= cfg.goal_y))


def is_fallen(st: State, cfg) -> torch.Tensor:
    """机身贴着**当地地形**了（相对高度，绝对阈值在台阶上永远触发不了）。"""
    return (st.base_pos[:, 2] - st.terrain_h) < cfg.fall_clearance


def is_flipped(st: State, cfg) -> torch.Tensor:
    rpy = st.rpy()
    return (torch.abs(rpy[:, 0]) > cfg.flip_rad) | (torch.abs(rpy[:, 1]) > cfg.flip_rad)


def is_out_of_course(st: State, cfg) -> torch.Tensor:
    """横向跑出地形（外面全是平地，可以绕过去白拿前进奖励）或倒退太多。"""
    return (torch.abs(st.base_pos[:, 1]) > cfg.out_y) | (st.base_pos[:, 0] < cfg.out_x_back)


def terminations(st: State, cfg, goal_z: float) -> dict[str, torch.Tensor]:
    """四个终止项的原始布尔量（Isaac 的 TerminationManager 一项一个 term）。

    成功和摔倒都是 `terminated`（不是 truncated）；超时单独走 `truncated`，
    这样 rsl_rl 才会用 `extras["time_outs"]` 正确地做 bootstrap。
    """
    success = is_success(st, cfg, goal_z)
    fallen = is_fallen(st, cfg)
    flipped = is_flipped(st, cfg)
    out = is_out_of_course(st, cfg)
    return {
        "success": success,
        "fallen": fallen,
        "flipped": flipped,
        "out_of_course": out,
        "any": success | fallen | flipped | out,
    }


# ------------------------------------------------------------------ 重置 / 命令


def sample_reset(cfg, terrain: CourseTerrain, n: int, generator=None) -> dict:
    """采样一批出生状态，分布与 `env.py:reset()` 完全一致。

    `z = top_footprint(x, y) + U(reset_z)`：home 关键帧的脚底本来就陷进地面约 18 mm，
    抬到地形上方让它自己落稳；**按整只狗的脚印取高度**，不能只用中心点
    （跨在台阶立面上时中心还在平地、前脚已探到台阶上方，见 AGENT.md §5-C1）。
    """
    device = terrain.boxes.device
    kw = {"generator": generator}

    def u(lo, hi, size=n):
        return torch.empty(size, dtype=F64, device=device).uniform_(lo, hi, **kw)

    x = u(*cfg.reset_x)
    y = u(-cfg.reset_y, cfg.reset_y)
    yaw = u(-cfg.reset_yaw, cfg.reset_yaw)
    z = terrain.top_footprint(x, y).to(F64) + u(*cfg.reset_z)
    return {"x": x, "y": y, "z": z, "yaw": yaw, "quat": yaw_to_quat(yaw)}


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """绕 z 转 yaw 的四元数 (w, x, y, z)，与 `env.py:172` 同式。"""
    half = yaw / 2.0
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


def sample_cmd(cfg, n: int, generator=None, device=None) -> torch.Tensor:
    """速度指令 U(cmd_vx)，**每个回合采一次、整回合不变**（对照 env.py:163）。"""
    return torch.empty(n, dtype=F64, device=device).uniform_(
        *cfg.cmd_vx, generator=generator)
