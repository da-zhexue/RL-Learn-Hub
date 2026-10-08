"""动作项：关节位置目标 = `q_default + action_scale × clip(action, -1, 1)`。

**为什么要自己包一层**：`env.py:194` 一进门就把动作 clamp 到 ±1，**而且 clamp 之后的值
才是后面观测和 `action_rate` 用的值**。Isaac Lab 的 `JointPositionAction` 默认不 clamp
（靠 cfg 里的 `clip` 字段，而那个字段 1.x 没有），策略吐出来的高斯样本落在 ±1 之外时
两边就不一样了——观测的第 45 维、`action_rate`、关节目标三处同时不同，还不会报错。

夹在 `process_actions` 里最省事：`super().process_actions` 会把参数原样存成
`self._raw_actions`，所以夹完再交给它，`raw_actions`（观测和 `action_rate` 读的）
和 `processed_actions`（写进仿真的关节目标）就都是夹过的值，一处生效、三处一致。

`process_actions` 这个方法名在 Isaac Lab 1.x/2.x 里都叫这个，是跨版本稳的。
"""
from __future__ import annotations

import torch

from isaaclab.utils import configclass

try:  # Isaac Lab 2.x
    from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
except ImportError:  # Isaac Lab 1.x
    from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class ClippedJointPositionAction(JointPositionAction):
    """把动作先夹到 ±1 再做位置目标（对照 `env.py:194-197`）。"""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(torch.clamp(actions, -1.0, 1.0))


@configclass
class ClippedJointPositionActionCfg(JointPositionActionCfg):
    """`env_cfg.py` 里挂它。其余字段（`scale` / `use_default_offset` / `joint_names`）照常。

    `@configclass` 照例不能少（理由见 `mdp/terrain.py:CourseTerrainCfg`）。本类只是覆盖
    基类已有字段，现在加不加都能跑；加上是为了以后往这里加字段时不必再踩一次坑。
    """

    class_type: type = ClippedJointPositionAction
