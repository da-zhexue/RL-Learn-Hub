"""rsl_rl 的训练超参配置（`Go2IsaacPPORunnerCfg`）。

只能上机 import（要 `isaaclab_rl`）。和算法无关的东西不在这里——
环境/奖励/时序都在 `go2_common/config.py`，两边共用。
"""
from go2_issac.agents.rsl_rl_ppo_cfg import (  # noqa: F401
    Go2IsaacPPORunnerCfg,
    iterations_for,
    mini_batches,
)

__all__ = ["Go2IsaacPPORunnerCfg", "iterations_for", "mini_batches"]
