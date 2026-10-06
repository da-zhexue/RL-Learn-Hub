"""诊断工具（PPO 版）：地形、步态、策略卡住时到底发生了什么。

    python3 -m go2_ppo.diag terrain     # 地形几何 & 课程缩放是否正确
    python3 -m go2_ppo.diag stance      # 标称站姿能站多高、肚子离地多少（决定爬不爬得过槛）
    python3 -m go2_ppo.diag lift        # 手调小跑的抬脚高度 -> 能跨多高的槛
    python3 -m go2_ppo.diag frontier    # 扫 action_scale × 槛高，量出可跨边界
    python3 -m go2_ppo.diag policy <model.zip> [地形] [scale]   # 训练后策略的抬脚高度
    python3 -m go2_ppo.diag why <model.zip> <scale>            # 回合为什么提前结束
    python3 -m go2_ppo.diag reward <model.zip> <scale>         # 分项奖励拆解

这 7 个子命令量的都是**环境**（地形几何、抬脚高度、奖励分项），与用哪个算法训练无关，
所以实现全在 `go2_sac.diag` 里、只有一份；这里只把它的 `load_policy` 换成 PPO 版再转发，
不复制那 400 多行。
"""

from __future__ import annotations

import pathlib
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import go2_sac.diag as _diag  # noqa: E402


def load_policy(path):
    from stable_baselines3 import PPO

    return PPO.load(path, device="cpu")


# 子命令里是 `load_policy(path)` 这种全局查找，所以在模块上换掉就够了
_diag.load_policy = load_policy
COMMANDS = _diag.COMMANDS


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
