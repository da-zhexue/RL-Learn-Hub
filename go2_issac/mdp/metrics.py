"""评估口径：和 MuJoCo 侧 `go2_common/train_utils.py:EvalMetricsCallback` **同名同义**。

跨仿真器的曲线只能比趋势，但"趋势"也得是同一个量才比得起来。所以这几个指标的定义
逐条照抄那边（连"跳过前 100 步再统计稳态速度"这种细节都一样）：

    eval/mean_reward   平均回合回报
    eval/success_rate  成功率
    eval/mean_level    平均爬到第几级台阶
    eval/max_level     最高到过第几级
    eval/mean_max_x    平均最远 x（`steps` 阶段没有真台阶，x 才是进度条）
    eval/mean_vx       稳态平均速度（跳过前 100 步的起步加减速）
    eval/vx_ratio      mean_vx / mean_cmd

**为什么不用 `MetricsManager`**：那是 Isaac Lab 2.1+ 才有的东西，而且它的 term 是逐环境
累积的，凑不出"最远 x / 跳过前 100 步"这种回合内量。这里改成包在 rsl_rl 外面的
`EpisodeTracker`——一圈 `step()` 正好是"一步"，累积时机没有任何歧义，
也不需要假设 Isaac 在哪一步调哪个 manager。

`EpisodeTracker` **故意不继承 rsl_rl 的 `VecEnvWrapper`**（那个类的 import 路径在 1.x/2.x
之间变过），只实现 `step()`，其余属性用 `__getattr__` 透传。rsl_rl 的 runner 需要的
`num_envs` / `num_actions` / `max_episode_length` / `device` / `reset()` 等等都会落到
里面的包装器上，路径差异就此躲开。
"""
from __future__ import annotations

from collections import deque

import torch

from go2_issac.mdp import state

#: 起步阶段不统计速度用的步数（对齐 `EvalMetricsCallback` 的 `step_count > 100`）
WARMUP_STEPS = 100

#: 必须和 MuJoCo 侧一一对应的记录名（`eval/<名字>` 两边的 TensorBoard 上都能直接对）
_FIELDS = ("mean_reward", "success_rate", "mean_level", "max_level",
           "mean_max_x", "mean_vx", "vx_ratio")


class EpisodeTracker:
    """累积最近 `window` 个回合的评估指标。包在 `RslRlVecEnvWrapper` 外面用。

        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
        env = EpisodeTracker(env, num_levels=len(course.stair_tops_for(...)))
        OnPolicyRunner(env, ...)
    """

    def __init__(self, vec_env, num_levels: int, window: int = 100, verbose: bool = True):
        self.vec_env = vec_env
        self.num_levels = int(num_levels)
        self.verbose = verbose
        self.window = int(window)
        self.done = deque(maxlen=self.window)
        #: 最近一次 `step()` 算出来的指标（`infos["log"]` 里也塞了一份，rsl_rl 会记进 TB）
        self.last: dict[str, float] = {}
        self._total_episodes = 0
        self._reset_buffers()

    # ---------------------------------------------------------- 透传

    def __getattr__(self, name):
        # 只有正常查找失败时才会进来；`vec_env` 本身在 __init__ 里已进实例字典，
        # 所以这里不会递归。dunder 直接放过（copy/pickle 会来看一眼）。
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.vec_env, name)

    # ---------------------------------------------------------- 每回合簿记

    def _reset_buffers(self, env_ids=None) -> None:
        base = self._base_env()
        n = base.num_envs
        dev = base.device
        if env_ids is None:
            self._max_x = torch.full((n,), -float("inf"), dtype=torch.float64, device=dev)
            self._rew_sum = torch.zeros(n, dtype=torch.float64, device=dev)
            self._vx_sum = torch.zeros(n, dtype=torch.float64, device=dev)
            self._cmd_sum = torch.zeros(n, dtype=torch.float64, device=dev)
            self._steady_n = torch.zeros(n, dtype=torch.float64, device=dev)
            self._level_max = torch.zeros(n, dtype=torch.long, device=dev)
            self._success = torch.zeros(n, dtype=torch.bool, device=dev)
            self._steps = torch.zeros(n, dtype=torch.long, device=dev)
            return
        self._max_x[env_ids] = -float("inf")
        self._rew_sum[env_ids] = 0.0
        self._vx_sum[env_ids] = 0.0
        self._cmd_sum[env_ids] = 0.0
        self._steady_n[env_ids] = 0.0
        self._level_max[env_ids] = 0
        self._success[env_ids] = False
        self._steps[env_ids] = 0

    # ---------------------------------------------------------- 给 play.py 用

    @property
    def episodes(self) -> int:
        """已归档的回合总数。`play.py` 靠它判断"这一步刚结束了一个回合"。"""
        return self._total_episodes

    @property
    def last_episode(self) -> dict | None:
        """最近结束的那个回合（窗口设 1 时就是它本身）。`step()` 之后取。"""
        return self.done[-1] if self.done else None

    def _base_env(self):
        """底下的 `ManagerBasedRLEnv`（`mdp.state` 要的就是它）。"""
        got = getattr(self.vec_env, "unwrapped", None)
        if got is not None and hasattr(got, "scene"):
            return got
        got = getattr(self.vec_env, "env", None)
        if got is not None and hasattr(got, "scene"):
            return got
        raise RuntimeError("找不到底下的 ManagerBasedRLEnv，EpisodeTracker 包错层了")

    def _record(self, done: torch.Tensor, rew: torch.Tensor) -> None:
        """把这一步的量并进"当前回合"，把结束掉的回合归档。"""
        base = self._base_env()
        st = state.get_state(base)

        self._rew_sum += rew.to(torch.float64)
        self._max_x = torch.maximum(self._max_x, st.base_pos[:, 0])
        self._level_max = torch.maximum(self._level_max, st.terrain_level)
        self._steps += 1
        steady = self._steps > WARMUP_STEPS
        self._vx_sum += torch.where(steady, st.base_lin_vel[:, 0], torch.zeros_like(self._vx_sum))
        self._cmd_sum += torch.where(steady, st.cmd_vx, torch.zeros_like(self._cmd_sum))
        self._steady_n += steady.to(torch.float64)
        self._success |= base.termination_manager.get_term("success")

        if bool(done.any()):
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            self._archive(ids)
            self._reset_buffers(ids)

    def _archive(self, env_ids: torch.Tensor) -> None:
        """一个回合结束：把这个回合的指标压进窗口。

        `steady_n` 可能为 0（回合在 100 步内就结束了，正是 `steps` 阶段卡在槛前的情形），
        这时退回用全回合的均值——和 `EvalMetricsCallback` 的 `short_episodes` 分支同款。
        """
        for e in env_ids.tolist():
            n_steady = float(self._steady_n[e])
            n_use = max(n_steady, float(self._steps[e]), 1.0)
            self.done.append({
                # 整个回合的回报之和（不是最后一步的）——对齐 `EvalMetricsCallback`
                # 里的 `rewards.append(total)`，`run_episode` 返回的就是累加值
                "reward": float(self._rew_sum[e]),
                "max_x": float(self._max_x[e]),
                "level": int(self._level_max[e]),
                "success": bool(self._success[e]),
                "vx": float(self._vx_sum[e] / n_use),
                "cmd": float(self._cmd_sum[e] / n_use),
                "steps": int(self._steps[e]),
            })
            self._total_episodes += 1

    # ---------------------------------------------------------- 对外

    def step(self, actions):
        obs, rew, dones, infos = self.vec_env.step(actions)
        self._record(torch.as_tensor(dones, dtype=torch.bool, device=rew.device), rew.detach())
        if len(self.done) >= max(1, self.window // 4):
            self.last = self.summarize()
            infos = dict(infos or {})
            infos["log"] = {f"eval/{k}": v for k, v in self.last.items()}
        return obs, rew, dones, infos

    def summarize(self) -> dict[str, float]:
        """窗口内的平均指标（字段名对齐 MuJoCo 侧，见模块 docstring）。"""
        d = list(self.done)
        if not d:
            return {}
        vx = float(sum(x["vx"] for x in d) / len(d))
        cmd = float(sum(x["cmd"] for x in d) / len(d))
        return {
            "mean_reward": float(sum(x["reward"] for x in d) / len(d)),
            "success_rate": float(sum(1.0 for x in d if x["success"]) / len(d)),
            "mean_level": float(sum(x["level"] for x in d) / len(d)),
            "max_level": float(max(x["level"] for x in d)),
            "mean_max_x": float(sum(x["max_x"] for x in d) / len(d)),
            "mean_vx": vx,
            "mean_cmd": cmd,
            "vx_ratio": vx / max(cmd, 1e-3),
        }

    def format_line(self) -> str:
        """一行给人看的摘要（`train.py` 定期打印，和 MuJoCo 侧那张表对得上）。"""
        s = self.last
        if not s:
            return ""
        return (f"回报 {s['mean_reward']:8.1f}  最远 x {s['mean_max_x']:+.2f}  "
                f"速度 {s['mean_vx']:+.2f}/{s['mean_cmd']:.2f} ({s['vx_ratio']:.2f}×)  "
                f"成功率 {s['success_rate'] * 100:5.1f}%  "
                f"最高台阶 {int(s['max_level'])}/{self.num_levels}")


def _self_check() -> None:
    """指标名字和 MuJoCo 侧必须一一对应（改了一个名字，两边的曲线就对不上）。

    去 `train_utils.py` 的**源码文本**里找 `eval/<名字>` 这串字符串——比"人肉记住"可靠，
    也比"在两边各留一份常量"少一处会漂移的地方。

    读文件而不是 `import go2_common.train_utils`：那个模块顶层就 `import mujoco` +
    `stable_baselines3`，而这段检查是要在**有显卡那台机器**上跑的，那边大概率没装这两个。
    grep 同样的字符串，证据一样，还不欠依赖。
    """
    import pathlib

    src_path = pathlib.Path(__file__).resolve().parents[2] / "go2_common" / "train_utils.py"
    src = src_path.read_text(encoding="utf-8")
    missing = [f for f in _FIELDS if f"eval/{f}" not in src]
    assert not missing, (
        f"`{src_path.name}` 里找不到这些记录名：{missing}；"
        f"两边的评估口径是一一对应的，改名字要一起改")


_self_check()
