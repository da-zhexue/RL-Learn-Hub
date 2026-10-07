"""ManagerBasedRLEnv 的各个 manager 用到的函数，一项一个薄壳，公式全在 `core.py`。

| 文件 | 挂到哪 | 里面是什么 |
|---|---|---|
| `state.py` | —— | 把 Isaac 的 articulation 数据折成 `core.State`（观测/奖励/终止唯一的取数口） |
| `observations.py` | `ObservationsCfg` | 45 维观测（**只有一个 term**，理由见文件头）+ "上一拍"的维护点 |
| `rewards.py` | `RewardsCfg` | 14 项奖励，一项一行转调 `core.term_*` |
| `terminations.py` | `TerminationsCfg` | 成功/摔倒/翻转/出界，转调 `core.is_*` |
| `events.py` | `EventCfg` | 重置出生点与关节初值（分布对齐 `env.py:reset()`） |
| `commands.py` | `CommandsCfg` | 只有 vx、只在重置时采样的 `CommandTerm` |
| `terrain.py` | `TerrainImporterCfg` | 盒子表 -> 高程图 patch |
| `metrics.py` | —— | 跨仿真器对比用的口径说明与断言 |

**这里所有东西都只能在有卡的机器上 import**（要 `isaaclab`）。本机跑的是
`course.py` / `core.py` 和 `tests/` 那两个对拍——它们不 import 这个包。
"""
from go2_issac.mdp import (  # noqa: F401
    commands,
    events,
    observations,
    rewards,
    state,
    terrain,
    terminations,
)

__all__ = [
    "commands", "events", "observations", "rewards", "state", "terrain", "terminations",
]
