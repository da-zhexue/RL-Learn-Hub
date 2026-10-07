"""速度指令：一个只有 vx、只在重置时采样的 `CommandTerm`。

**为什么不用 Isaac Lab 自带的 `UniformVelocityCommand`**：
  1. 它按 `resampling_time_range` 在回合中途重采样，而 `env.py:163` 是**每回合采一次、
     整回合不变**（改指令会让 `track_lin_vel` 的对比基准中途跳变，学的行为不一样）；
  2. 它的指令是**机体系**的（`lin_vel_b`），我们这边 `track_lin_vel` 用的是**世界系** vx
     （`env.py` 的 `r.base_v[0]`），地形关于 y 对称、按 x 前进，世界系才对得上；
  3. 它还会发 yaw 指令，我们不发（朝向由 `reward_yaw` 管）。
自己写一个只有 20 行的 CommandTerm，比配一堆 `ranges` 去把它调成我们的样子清楚得多。

指令采样用 `core.sample_cmd`，和 MuJoCo 侧同一个分布 `EnvCfg.cmd_vx = (0.4, 0.8)`。
"""
from __future__ import annotations

import torch

from go2_common.config import EnvCfg
from go2_issac import core
from go2_issac.mdp import state

try:  # Isaac Lab 2.x
    from isaaclab.envs.mdp import CommandTerm, CommandTermCfg
except ImportError:  # Isaac Lab 1.x
    from isaaclab.managers import CommandTerm, CommandTermCfg

F64 = torch.float64


class CourseVelocityCommand(CommandTerm):
    """每回合一个 `cmd_vx ~ U(0.4, 0.8)`，整回合不变。"""

    cfg: "CourseVelocityCommandCfg"

    def __init__(self, cfg: "CourseVelocityCommandCfg", env):
        super().__init__(cfg, env)
        # (N, 1)：CommandManager 期望 command 的第一维是环境数；
        # 通道数 1 表示"只有前进速度"，不要照抄 Isaac 惯用的 3（vx, vy, wz）。
        self._command = torch.zeros(env.num_envs, 1, dtype=torch.float32, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    # -------------------------------------------------------------- 采样

    def _env_cfg(self) -> EnvCfg:
        """`go2_common` 的环境配置（`cmd_vx` 分布从那里读，不抄数字）。"""
        try:
            return state._ctx(self._env).cfg
        except RuntimeError:
            # ctx 还没建：`CommandManager.__init__` 在 `event_manager.apply("startup")`
            # 之前就会先 sample 一次。`EnvCfg()` 的默认值就是训练用的那一套，
            # 而 `cmd_vx` 的分布不受 `--terrain/--terrain-scale` 影响，所以等价。
            return EnvCfg()

    def _resample_command(self, env_ids):
        n = len(env_ids)
        sample = core.sample_cmd(self._env_cfg(), n, device=self.device)
        self._command[env_ids, 0] = sample.to(self._command.dtype)

    def _update_command(self) -> None:
        """每步都调；什么都不用做（指令整回合不变）。"""

    # -------------------------------------------------------------- 指标
    #
    # **这只是个"瞬时视角"的辅助通道**：`CommandManager` 会把这些按
    # `Metrics/command/<term>/<key>` 写进 `extras["log"]`，rsl_rl 记到 TensorBoard。
    # 权威的评估口径是 `mdp/metrics.py:EpisodeTracker`（按回合累积，逐项对齐
    # `EvalMetricsCallback`）——这里给的是"这一步所有环境上的均值"，两条曲线
    # 数值不会完全一样，看趋势用哪个都行。
    #
    # `*args` 是刻意的：`_update_metrics()` 的签名在 Isaac Lab 各版本之间变过
    # （有的带 `update_interval` 参数），多收参数比赌签名安全。

    def _update_metrics(self, *args, **kwargs) -> None:
        try:
            ctx = state._ctx(self._env)
        except RuntimeError:
            # `setup()` 还没跑（CommandManager 先于 startup 事件构造）
            return
        st = state.get_state(self._env)
        term = core.terminations(st, ctx.cfg, ctx.goal_z)

        # 本步各量在所有环境上的均值；`CommandManager` 会按平均回合长度归一化。
        self.metrics["success"] = term["success"].float().mean()
        self.metrics["fallen"] = term["fallen"].float().mean()
        self.metrics["level"] = st.terrain_level.float().mean()
        # 速度跟踪比：|vx| / cmd_vx，1.0 表示跟得上，<1 表示磨蹭
        self.metrics["vx_ratio"] = (
            st.base_lin_vel[:, 0].abs() / st.cmd_vx.clamp(min=1e-3)).mean()

    # -------------------------------------------------------------- 可视化占位

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """不做可视化（要看指令就用 TensorBoard 的 `Metrics/command/...`）。"""

    def _debug_vis_callback(self, event) -> None:
        """同 `_set_debug_vis_impl`：这里没有 marker 要更新。"""


class CourseVelocityCommandCfg(CommandTermCfg):
    """`env_cfg.py` 里挂它。`class_type` + `resampling_time_range` 是 Isaac 的约定写法。"""

    class_type: type = CourseVelocityCommand

    # 基类要求给这个字段（默认是 MISSING）；给一个"永远不重采样"的区间：
    # 指令只在重置时采（理由见模块 docstring 第 1 条）。
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
