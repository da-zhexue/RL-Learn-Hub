"""观测：直接调 `core.obs`，**只有一个 ObsTerm**。

为什么不像 Isaac Lab 的惯用法那样拆成 6 个 term（陀螺 / 投影重力 / 指令 / 关节角 / 关节速 /
上一步动作）：那 6 段的缩放（`OBS_GYRO_SCALE=0.25`、`OBS_DQ_SCALE=0.05`、`OBS_CMD_SCALE=2.0`、
clip ±10）**已经**在 `go2_common/config.py` 里定死、并被 `core.obs` 和 MuJoCo 侧共用。
拆成 6 个 Isaac term 就得在 `env_cfg.py` 里再抄一遍这些数——两份配置漂移正是这次迁移要
避免的事（`tests/core_parity.py` 对的是 `core.obs`，不是 Isaac 的 term 表）。
所以这里保持"一段进、一段出"，让 45 维的语义只有一份实现。

**这一步还是"上一拍"唯一的维护点**（`_advance_prev`）。理由：`ManagerBasedRLEnv.step()`
里观测是**最后**跑的（终止 -> 奖励 -> 重置 -> 命令 -> 观测），所以在这里推进
`env._course_prev`，本步的终止/奖励就已经用上了旧的 `prev_*`，而下一步会读到这里写的新值——
与 `env.py:245-248`"先用旧值算奖励、再推进"的顺序等价。放到别处（比如奖励项里）都会踩到
"同一个对象被重复推进"或"推进得太早"。

**和 MuJoCo 的一处已知差异**：重置后的第一步，`prev_action` 在 MuJoCo 里是 0
（`env.py:175` 显式清零），这里是重置前最后一次下发的动作。两者各自自洽，只影响
重置后第一步 `action_rate` 的取值（权重 -0.05，量级 ~1），不影响别的项。
"""
from __future__ import annotations

import torch

from go2_issac import core
from go2_issac.mdp import state

F64 = torch.float64


def policy_obs(env) -> torch.Tensor:
    """45 维（`--privileged` 时 46 维）观测，float32、clip ±10。"""
    ctx = state._ctx(env)
    st = state.get_state(env)
    out = core.obs(st, ctx.cfg, ctx.q_default, privileged=ctx.cfg.privileged)
    _advance_prev(env, st)
    return out


def _advance_prev(env, st: core.State) -> None:
    """把这一拍的量留给下一拍（顺序见模块 docstring）。"""
    env._course_prev = core.advance(st)


def init_prev(env, env_ids=None) -> None:
    """`mode="startup"` 事件：把 `_course_prev` 填上初值。

    `env.py:72-74` 的初值就是全零（`prev_action=0, prev_x=0, prev_terrain_h=0, level=0`），
    照抄。真正的值在第一次 `policy_obs` 之后就有了。
    """
    n = env.scene.env_origins.shape[0]
    device = env.device
    env._course_prev = {
        "prev_action": torch.zeros(n, 12, dtype=F64, device=device),
        "prev_x": torch.zeros(n, dtype=F64, device=device),
        "prev_terrain_h": torch.zeros(n, dtype=F64, device=device),
        "prev_terrain_level": torch.zeros(n, dtype=torch.long, device=device),
    }
