"""两个训练入口（SAC / PPO）共用的脚手架：环境工厂、配置还原、评估回调。

这里的每一件都与算法无关——`EvalMetricsCallback` 只调 `model.predict()`，
`load_resume_configs` 只读 `config.json`——所以只有一份，算法包里不再各抄一遍。
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import fields

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from go2_common.config import EnvCfg, RewardCfg
from go2_common.env import Go2TerrainEnv, run_episode

CONFIG_NAME = "config.json"


def make_env(env_cfg: EnvCfg, reward_cfg: RewardCfg, seed: int):
    """环境工厂。必须是模块级函数并配 functools.partial —— SubprocVecEnv 要 pickle 它，
    闭包 lambda 容易在这里翻车。"""
    env = Go2TerrainEnv(env_cfg, reward_cfg)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


class EvalMetricsCallback(BaseCallback):
    """定期用确定性策略跑几个完整回合，记录成功率 / 爬到第几级台阶 / 平均回报。

    这里没有用 SB3 自带的 EvalCallback：它只会记回报，而我们要的指标
    （成功率、最高台阶）在 info 里。自己起一个**非向量化**的环境跑，最简单可靠。
    """

    def __init__(self, env_cfg, reward_cfg, eval_freq, n_episodes, save_dir, verbose=0):
        super().__init__(verbose)
        self.eval_env = Go2TerrainEnv(env_cfg, reward_cfg)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.save_dir = pathlib.Path(save_dir)
        self.best_reward = -np.inf
        self.last_eval = 0

    def _act(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def _on_step(self) -> bool:
        # 用 num_timesteps（总转移数）而不是 n_calls（向量环境步数），前者才是"训练步数"
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps

        rewards, levels, max_xs, n_success = [], [], [], 0
        vxs, cmds = [], []
        vxs_all, cmds_all = [], []
        for i in range(self.n_episodes):
            # 记每个回合到过的最远 x：steps 阶段没有真台阶（只有两个 0.08 m 的槛），
            # 光看"最高台阶 0/0"和成功率看不出任何进展，x 才是那个阶段的进度条。
            best_x = [-np.inf]

            def on_step(e):
                best_x[0] = max(best_x[0], float(e.data.qpos[0]))
                vx_now, cmd_now = float(e.data.qvel[0]), float(e.cmd_vx)
                vxs_all.append(vx_now)
                cmds_all.append(cmd_now)
                # 跳过前 100 步（起步 2 s 的加减速），只统计稳态（判据见 AGENT.md §5-D5）
                if e.step_count > 100:
                    vxs.append(vx_now)
                    cmds.append(cmd_now)

            total, info = run_episode(self.eval_env, self._act, seed=10_000 + i, on_step=on_step)
            rewards.append(total)
            levels.append(info["level"])
            max_xs.append(best_x[0])
            n_success += int(info["success"])

        # 回合全在 100 步内结束时 vxs 是空的（np.mean([]) = NaN），退回用全部步的数据，
        # 并在打印时标出来（见 AGENT.md §5-D4）。
        short_episodes = not vxs
        if short_episodes:
            vxs, cmds = vxs_all, cmds_all
        mean_reward = float(np.mean(rewards))
        mean_vx, mean_cmd = float(np.mean(vxs)), float(np.mean(cmds))
        vx_ratio = mean_vx / max(mean_cmd, 1e-3)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/success_rate", n_success / self.n_episodes)
        self.logger.record("eval/mean_level", float(np.mean(levels)))
        self.logger.record("eval/max_level", float(np.max(levels)))
        self.logger.record("eval/mean_max_x", float(np.mean(max_xs)))
        self.logger.record("eval/mean_vx", mean_vx)
        self.logger.record("eval/vx_ratio", vx_ratio)
        if self.verbose:
            print(
                f"  [eval @ {self.num_timesteps:>9,} 步] 回报 {mean_reward:8.1f}  "
                f"最远 x {np.mean(max_xs):+.2f}  "
                f"速度 {mean_vx:+.2f}/{mean_cmd:.2f} ({vx_ratio:.2f}×)  "
                f"成功率 {n_success}/{self.n_episodes}  "
                f"最高台阶 {int(np.max(levels))}/{len(self.eval_env.terrain.stair_tops)}"
                + ("  ⚠回合全在100步内结束" if short_episodes else "")
            )

        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_dir / "best_model.zip"))
        return True


def resolve_model_path(path) -> pathlib.Path:
    """把用户给的模型路径补全成实际存在的 .zip。

    SAC.load 只在"路径本身没写后缀"时才帮忙补 .zip；写了 `.../rl_1000000_steps` 这种
    没后缀的文件名时它会直接去找 `rl_1000000_steps.zip` 然后报 FileNotFoundError。
    """
    p = pathlib.Path(path)
    if p.exists():
        return p
    with_zip = pathlib.Path(str(p) + ".zip")
    return with_zip if with_zip.exists() else p


def load_resume_configs(model_path: pathlib.Path):
    """从 checkpoint 同目录（或其上一层）的 config.json 还原 (EnvCfg, RewardCfg)。

    否则很容易出现"用 45 维的配置去加载 46 维的模型"这种事。
    往上找一层是因为 CheckpointCallback 存到 `checkpoints/` 子目录里，而 config.json
    在运行目录的根上（`models/go2_sac_flat_xxx/config.json`）。
    """
    candidates = [model_path.parent / CONFIG_NAME, model_path.parent.parent / CONFIG_NAME]
    cfg_path = next((c for c in candidates if c.exists()), None)
    if cfg_path is None:
        print(f"  ! {candidates[0]} 不存在，改用命令行参数构造环境（维度对不上会在 load 时报错）")
        return None, None
    with open(cfg_path, encoding="utf-8") as f:
        saved = json.load(f)
    env_cfg = EnvCfg(**{k: v for k, v in saved["env"].items() if k in {f.name for f in fields(EnvCfg)}})
    reward_cfg = RewardCfg(
        **{k: v for k, v in saved["reward"].items() if k in {f.name for f in fields(RewardCfg)}}
    )
    print(f"  已从 {cfg_path} 还原环境配置：terrain={env_cfg.terrain} privileged={env_cfg.privileged}")
    return env_cfg, reward_cfg
