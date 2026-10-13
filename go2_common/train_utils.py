"""两个训练入口（SAC / PPO）共用的脚手架：环境工厂、配置还原、评估回调。

这里的每一件都与算法无关——`EvalMetricsCallback` 只调 `model.predict()`，
`load_resume_configs` 只读 `config.json`——所以只有一份，算法包里不再各抄一遍。
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import fields, replace
from functools import partial

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from go2_common.config import EnvCfg, RewardCfg
from go2_common.env import Go2TerrainEnv, run_episode

CONFIG_NAME = "config.json"
BEST_METRIC_NAME = "best_metric.json"  # best_model 的选模 key，格式见 best_metric_payload
EVAL_SEED_BASE = 10_000  # 评估回合的固定种子，**不要改**：改了历史 best 的 key 就不可比了


def task_key(n_success, mean_level, mean_max_x, mean_reward, goal_x) -> tuple:
    """选模判据：字典序 (成功数, 峰值台阶数, 最远 x, 回报)。

    **为什么回报不能单独当判据**（这是实测出来的、不是理论担心）：full 阶段
    （runs/go2_sac_full/SAC_0）回报从 -13.86 涨到 +172.30 的那 20 次评估里，成功率恒为 0、
    峰值台阶数从 0.15 掉到 0.05。涨的全是"少挨罚"（碰撞项 -1951 → -110 / 20 局），
    不是"多爬一级"。按回报取极大值最终选中的是 rl_2700032：冲 1.8 m 就地翻滚
    （20/20 回合摔倒，+172.3），而同一批评估里真有进展的那次（2,051,232 步，峰值 0.15）
    回报只有 -29.92，永远选不上 —— 于是"模型退化"被 best_model 一个文件忠实地记录了下来。

    四项的含义：
      n_success    固定 20 局里成功的局数 —— 任务本身。任何 >0 都严格压过所有 0，
                   这是这条 key 存在的全部理由（§3 规则 7 的成功判据一个字没动）。
      mean_level   轨迹峰值台阶数（含出生姿态）—— 成功之外的"最多爬到第几级"。
      mean_max_x   最远 x，**截到 goal_x**。截断是为了掐掉 §5-B5 那条"绕到 |y|≈2.2 的
                   平地上一路向前"的刷分路：那里 x 可以很大，但一级台阶也没爬。
      mean_reward  只在上面三项全同时才比 —— 兜底。没有它，同一层级的两份权重分不出高低。

    比较必须用严格 `>`：评估是**固定种子 10000+i 的确定性配对比较**（实测同一份权重连跑两次
    数字逐位相同），所以"平局"就是同一水平，不该反复覆盖 best_model 去刷磁盘。
    """
    # 逐项 round：key 要写进 best_metric.json 再读回来比大小，浮点尾数不该参与比较
    return (
        int(n_success),
        round(float(mean_level), 2),
        round(float(min(mean_max_x, goal_x)), 2),
        round(float(mean_reward), 1),
    )


def best_metric_payload(key, metrics, num_timesteps, protocol) -> dict:
    """best_metric.json 的内容。除了 key 本身，把**用哪套口径量出来的**一起存下来：
    换了地形尺度/动作幅度以后，老 key 就不可比了，必须能检测出来（见 _bootstrap_best）。"""
    return {
        "key": list(key),
        "success": int(metrics["n_success"]),
        "mean_level": float(metrics["mean_level"]),
        "mean_max_x": float(metrics["mean_max_x"]),
        "mean_reward": float(metrics["mean_reward"]),
        "step": int(num_timesteps),
        "protocol": dict(protocol),
    }


def worker_scale(env_cfg: EnvCfg, seed: int) -> float:
    """这个 worker 该用哪个 terrain_scale。

    `terrain_scale_range` 为 None（默认）时原样返回 `cfg.terrain_scale`，**不消耗任何
    RNG、不做任何浮点运算**，所以旧行为的逐位可复现性不受影响。

    区间给定时用**黄金比低差异序列**分配，不用随机抽签：worker 数 8 时 8 个采样点在整个
    区间上近似等距（随机抽签会挤堆，可能 8 个点全落在半边上）。而且同一个 seed 永远给同一个
    scale —— 换 seed 换布局、固定 seed 完全可复现。
    """
    rng_range = getattr(env_cfg, "terrain_scale_range", None)
    if not rng_range:
        return float(env_cfg.terrain_scale)
    lo, hi = (float(v) for v in rng_range)
    if not (0.0 <= lo <= hi):
        raise ValueError(f"terrain_scale_range 必须是 0 <= lo <= hi，收到 {rng_range!r}")
    u = (int(seed) * 0.6180339887498949) % 1.0  # 1/φ，连续整数 seed 的相位均匀铺开
    return lo + (hi - lo) * u


def make_env(env_cfg: EnvCfg, reward_cfg: RewardCfg, seed: int):
    """环境工厂。必须是模块级函数并配 functools.partial —— SubprocVecEnv 要 pickle 它，
    闭包 lambda 容易在这里翻车。"""
    scale = worker_scale(env_cfg, seed)
    if scale != env_cfg.terrain_scale:
        # 必须走 replace()：Go2TerrainEnv.__init__ 会把 cfg.goal_z 改写成本地形算出来的值
        # （env.py），共用一份对象会让各 worker 互相串。
        env_cfg = replace(env_cfg, terrain_scale=scale)
    env = Go2TerrainEnv(env_cfg, reward_cfg)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def _policy_act(policy, obs, env):
    """确定性策略选一个动作。run_episode 要的是 act(obs, env)，所以外面套 partial。"""
    action, _ = policy.predict(obs, deterministic=True)
    return action


class EvalMetricsCallback(BaseCallback):
    """定期用确定性策略跑几个完整回合，记录成功率 / 爬到第几级台阶 / 平均回报。

    这里没有用 SB3 自带的 EvalCallback：它只会记回报，而我们要的指标
    （成功率、最高台阶）在 info 里。自己起一个**非向量化**的环境跑，最简单可靠。

    两件容易踩的事都在这里处理掉了：
      1. **best_model 按 task_key 的字典序选，不按回报**（理由见 task_key 的注释）。
      2. 每次评估额外在若干**参考尺度**上各跑一遍（如 0.56 / 0.70）。实测：full 阶段
         95 万步把 0.56 尺度的峰值台阶数从 0.95 砍到 0.45、最终台阶数从 0.75 砍到 0.15，
         而当时记录的**每一个**指标都还在变好 —— 只在当前尺度上看，看不出这种回退。
         参考尺度的指标只进 TensorBoard / eval_metrics.jsonl，**永远不参与选模**：
         否则策略会去刷低台阶的分而不是完成当前尺度的任务。
    """

    def __init__(self, env_cfg, reward_cfg, eval_freq, n_episodes, save_dir, verbose=0, save_buffer=False,
                 *, reference_scales=(), reference_episodes=None, seed_base=EVAL_SEED_BASE):
        # 后三个是 keyword-only：go2_ppo/train.py 和测试里都是按位置传前五个的
        # （见 AGENT.md §5-F1：改签名时最容易在这里出事）。
        super().__init__(verbose)
        # 评估环境一律**固定远端出生**（reset_frac_stair=0）：选模 key 和课程闸门都是跨
        # 阶段/跨运行的比较量，出生分布一变，历史 key 就不可比了。
        # 每个 env 各拿一份 replace() 出来的配置副本 —— Go2TerrainEnv.__init__ 会把
        # cfg.goal_z 改写成按地形算出来的值（env.py），共用一份对象会互相串。
        def _mk(scale):
            cfg = replace(env_cfg)
            cfg.reset_frac_stair = 0.0
            # 训练侧的 worker 尺度是 train_utils.worker_scale 在 make_env 里现算的，
            # 根本不过 EnvCfg.terrain_scale_range 这条路；这里显式清掉只是防线：
            # 将来要是把混合搬到 env.reset() 里逐回合抽（per-episode 重建），
            # 忘了清就会让评估尺子随回合乱跳，选模 key 静默失去可比性。
            cfg.terrain_scale_range = None
            cfg.terrain_scale = float(scale)
            return Go2TerrainEnv(cfg, reward_cfg)

        self.eval_env = _mk(env_cfg.terrain_scale)
        # 和当前尺度相同的参考尺度没有信息量，只是白跑一遍（一个尺度约 12.6 s）：
        # 课程梯子上每一档都配 (0.56, 0.70) 时，0.56 那一档就正好撞上。
        self.reference_envs = {
            float(s): _mk(s) for s in reference_scales
            if abs(float(s) - float(env_cfg.terrain_scale)) > 1e-9
        }
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.reference_episodes = int(reference_episodes or n_episodes)
        self.seed_base = int(seed_base)
        self.save_dir = pathlib.Path(save_dir)
        self.best_key = None  # None = 还没有基准，第一次评估一定存
        self.last_eval = 0
        self.save_buffer = save_buffer

    # ------------------------------------------------------------------ 口径

    def _protocol(self) -> dict:
        """这套 key 是在什么口径下量出来的。只有这几项全一样，key 之间才能比大小。"""
        cfg = self.eval_env.cfg
        return {
            "terrain": str(cfg.terrain),
            "terrain_scale": float(cfg.terrain_scale),
            "action_scale": float(cfg.action_scale),
            "n_episodes": int(self.n_episodes),
            "seed_base": int(self.seed_base),
            "deterministic": True,
            # 观测维度也要进口径（2026-10-13 加）。原来没有它，于是"47 维量出来的 key"
            # 和"63 维量出来的 key"会被判成同一口径直接比大小 —— 而开了地形扫描之后
            # 两者根本不是同一个策略。加进来之后这类跨越维度的比较会自动触发重新立基准。
            "obs_dim": int(self.eval_env.observation_space.shape[0]),
        }

    def _evaluate(self, env, n_episodes, policy=None) -> dict:
        """跑 n_episodes 个确定性回合，返回指标 dict。env 由调用方给（当前尺度或参考尺度）。"""
        policy = self.model if policy is None else policy
        act = partial(_policy_act, policy)
        rewards, levels, final_levels, max_xs, n_success = [], [], [], [], 0
        vxs, cmds = [], []
        vxs_all, cmds_all = [], []
        for i in range(n_episodes):
            # 记每个回合到过的最远 x：steps 阶段没有真台阶（只有两个 0.08 m 的槛），
            # 光看"最高台阶 0/0"和成功率看不出任何进展，x 才是那个阶段的进度条。
            best_x = [-np.inf]
            best_level = [0]

            def on_step(e):
                best_x[0] = max(best_x[0], float(e.data.qpos[0]))
                best_level[0] = max(best_level[0], e.support_level())
                vx_now, cmd_now = float(e.data.qvel[0]), float(e.cmd_vx)
                vxs_all.append(vx_now)
                cmds_all.append(cmd_now)
                # 跳过前 100 步（起步 2 s 的加减速），只统计稳态（判据见 AGENT.md §5-D5）
                if e.step_count > 100:
                    vxs.append(vx_now)
                    cmds.append(cmd_now)

            total, info = run_episode(env, act, seed=self.seed_base + i, on_step=on_step)
            rewards.append(total)
            # Include the reset pose, even if the first step immediately descends.
            levels.append(max(best_level[0], info["max_level"]))
            final_levels.append(info["level"])
            max_xs.append(best_x[0])
            n_success += int(info["success"])

        # 回合全在 100 步内结束时 vxs 是空的（np.mean([]) = NaN），退回用全部步的数据，
        # 并在打印时标出来（见 AGENT.md §5-D4）。
        short_episodes = not vxs
        if short_episodes:
            vxs, cmds = vxs_all, cmds_all
        mean_vx, mean_cmd = float(np.mean(vxs)), float(np.mean(cmds))
        return {
            "n_episodes": int(n_episodes),
            "n_success": int(n_success),
            "success_rate": n_success / n_episodes,
            "mean_reward": float(np.mean(rewards)),
            "mean_level": float(np.mean(levels)),
            "max_level": float(np.max(levels)),
            "mean_final_level": float(np.mean(final_levels)),
            "max_final_level": float(np.max(final_levels)),
            "mean_max_x": float(np.mean(max_xs)),
            "mean_vx": mean_vx,
            "mean_cmd": mean_cmd,
            "vx_ratio": mean_vx / max(mean_cmd, 1e-3),
            "short_episodes": short_episodes,
            "stair_levels": len(env.terrain.stair_tops),
        }

    # ------------------------------------------------------------------ best_model

    def _bootstrap_best(self) -> None:
        """续训/换阶段时先把基准立起来，否则第一次评估就会盖掉继承来的 best_model。

        两种情况：
          - 同目录已有 best_metric.json 且口径一致 → 直接读 key（免费，不跑评估）；
          - 没有可读的基准、但有 best/best_model.zip（换过地形尺度/动作幅度的续训）
            → 用**当前口径**把它重新量一遍当基准，这样"这一阶段一步没进步，就原样
            继承上一个阶段的 best_model"是真的。
        两次都落空（全新运行目录）→ 保持 None，第一次评估直接存 —— 新目录没有任何
        "更好的东西"需要保护，而且这样能保证 best/best_model.zip 一定会被写出来。
        """
        metric_path = self.save_dir / BEST_METRIC_NAME
        protocol = self._protocol()
        saved = None
        if metric_path.exists():
            try:
                saved = json.loads(metric_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = None
            if saved and saved.get("protocol") == protocol:
                self.best_key = tuple(saved["key"])
                print(f"  选模基准沿用 {metric_path}：key={tuple(saved['key'])}（step {saved.get('step')}）")
                return
            if saved:
                print(f"  ! {metric_path} 的口径与当前不一致（{saved.get('protocol')} != {protocol}），重新立基准")

        inherited = self.save_dir / "best_model.zip"
        if not inherited.exists():
            return
        try:
            from stable_baselines3 import SAC  # 本地导入：只有走这条路才需要

            policy = SAC.load(str(inherited), device="cpu")
        except Exception as exc:  # noqa: BLE001 —— 老维度/老算法的模型加载不了是常事
            print(f"  ! 继承的 {inherited} 加载失败（{exc}），本次以第一次评估为基准")
            return
        metrics = self._evaluate(self.eval_env, self.n_episodes, policy=policy)
        self.best_key = task_key(metrics["n_success"], metrics["mean_level"], metrics["mean_max_x"],
                                 metrics["mean_reward"], self.eval_env.cfg.goal_x)
        print(f"  选模基准取自继承的 {inherited}：key={self.best_key}")
        self._write_best_metric(self.best_key, metrics)

    def _write_best_metric(self, key, metrics) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / BEST_METRIC_NAME).write_text(
            json.dumps(best_metric_payload(key, metrics, self.num_timesteps, self._protocol()),
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _init_callback(self) -> None:
        # SB3 在 learn() 开头调一次，此时 self.model 已就位（见 AGENT.md §5-F1）。
        # 放这里而不是 __init__：构造回调不该偷偷跑一分钟的评估。
        self._bootstrap_best()

    # ------------------------------------------------------------------ 主循环

    def _on_step(self) -> bool:
        # 用 num_timesteps（总转移数）而不是 n_calls（向量环境步数），前者才是"训练步数"
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps

        metrics = self._evaluate(self.eval_env, self.n_episodes)
        self.logger.record("eval/mean_reward", metrics["mean_reward"])
        self.logger.record("eval/success_rate", metrics["success_rate"])
        self.logger.record("eval/mean_level", metrics["mean_level"])
        self.logger.record("eval/max_level", metrics["max_level"])
        self.logger.record("eval/mean_final_level", metrics["mean_final_level"])
        self.logger.record("eval/max_final_level", metrics["max_final_level"])
        self.logger.record("eval/mean_max_x", metrics["mean_max_x"])
        self.logger.record("eval/mean_vx", metrics["mean_vx"])
        self.logger.record("eval/vx_ratio", metrics["vx_ratio"])

        ref_metrics = {}
        for scale, env in self.reference_envs.items():
            ref = self._evaluate(env, self.reference_episodes)
            ref_metrics[scale] = ref
            tag = f"eval/ref_{int(round(scale * 100)):03d}"
            self.logger.record(f"{tag}/mean_reward", ref["mean_reward"])
            self.logger.record(f"{tag}/success_rate", ref["success_rate"])
            self.logger.record(f"{tag}/mean_level", ref["mean_level"])
            self.logger.record(f"{tag}/max_level", ref["max_level"])
            self.logger.record(f"{tag}/mean_final_level", ref["mean_final_level"])
            self.logger.record(f"{tag}/mean_max_x", ref["mean_max_x"])
            self.logger.record(f"{tag}/vx_ratio", ref["vx_ratio"])

        key = task_key(metrics["n_success"], metrics["mean_level"], metrics["mean_max_x"],
                       metrics["mean_reward"], self.eval_env.cfg.goal_x)
        improved = self.best_key is None or key > tuple(self.best_key)
        self.logger.record("best/success_rate", metrics["success_rate"])
        self.logger.record("best/mean_level", metrics["mean_level"])
        self.logger.record("best/mean_max_x", metrics["mean_max_x"])
        self.logger.record("best/mean_reward", metrics["mean_reward"])
        self.logger.record("best/updated", 1.0 if improved else 0.0)

        if self.verbose:
            print(
                f"  [eval @ {self.num_timesteps:>9,} 步] 回报 {metrics['mean_reward']:8.1f}  "
                f"最远 x {metrics['mean_max_x']:+.2f}  "
                f"速度 {metrics['mean_vx']:+.2f}/{metrics['mean_cmd']:.2f} "
                f"({metrics['vx_ratio']:.2f}×)  "
                f"成功率 {metrics['n_success']}/{self.n_episodes}  "
                f"最高台阶 {int(metrics['max_level'])}/{metrics['stair_levels']}  "
                f" peak {metrics['mean_level']:.2f} / final {metrics['mean_final_level']:.2f}"
                + ("  ⚠回合全在100步内结束" if metrics["short_episodes"] else "")
                + (f"  ← best key {key}" if improved else "")
            )
            for scale, ref in sorted(ref_metrics.items()):
                print(
                    f"      ↳ 参考尺度 {scale:.2f}：成功率 {ref['n_success']}/{ref['n_episodes']}  "
                    f"peak {ref['mean_level']:.2f}  final {ref['mean_final_level']:.2f}  "
                    f"最远 x {ref['mean_max_x']:+.2f}"
                )

        if improved:
            self.best_key = key
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if self.save_buffer:
                self.model.save_replay_buffer(str(self.save_dir / "best_model.pkl")) # 方便加载取相同名字，实际best_model.pkl为replay buffer
            self.model.save(str(self.save_dir / "best_model.zip"))
            self._write_best_metric(key, metrics)

        self._log_metrics_json(metrics, ref_metrics, key, improved)
        return True

    def _log_metrics_json(self, metrics, ref_metrics, key, improved) -> None:
        """每次评估追加一行 JSON 到 `<save_dir>/../eval_metrics.jsonl`，供课程闸门与复盘读。

        为什么不只写 TensorBoard：闸门判据是**参考尺度的成功率**加上当前尺度的进展，
        要按"这一次评估"整体判读，逐标量的 TB 记录反而不好用。一行一个 JSON 对象，
        追加写、不重写，崩了也不丢历史。
        """
        try:
            path = self.save_dir.parent / "eval_metrics.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "step": int(self.num_timesteps),
                "protocol": self._protocol(),
                "full": {k: metrics[k] for k in
                         ("n_success", "n_episodes", "success_rate", "mean_reward", "mean_level",
                          "max_level", "mean_final_level", "mean_max_x", "vx_ratio")},
                "ref": {f"{s:.2f}": {k: m[k] for k in
                                     ("n_success", "n_episodes", "success_rate", "mean_reward",
                                      "mean_level", "max_level", "mean_final_level", "mean_max_x", "vx_ratio")}
                        for s, m in ref_metrics.items()},
                "key": list(key),
                "best_updated": bool(improved),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:  # 磁盘满过一次了（日志.md 10.11），这里绝不能让训练挂掉
            print(f"  ! 写 {self.save_dir.parent / 'eval_metrics.jsonl'} 失败：{exc}")


def find_replay_buffer(model_path: pathlib.Path):
    """找模型旁边的 replay buffer，没有就返回 None。

    两种命名都认：
      - 与模型同名：`best_model.zip` / `model.zip` 旁边的 `best_model.pkl` / `model.pkl`
      - CheckpointCallback 的：`rl_1000000_steps.zip` 旁边的 `rl_replay_buffer_1000000_steps.pkl`
        （SB3 在前缀后面插了 "replay_buffer_"，见 callbacks.py 的 `_checkpoint_path`）
    第二种按"步数后缀"去 glob，不硬拼名字——`name_prefix` 里可能还有下划线。
    """
    same_name = pathlib.Path(str(model_path)[: -len(".zip")] + ".pkl")
    if same_name.exists():
        return same_name
    steps_tail = "_".join(model_path.stem.rsplit("_", 2)[-2:])  # "1000000_steps"
    hits = sorted(model_path.parent.glob(f"*_replay_buffer_{steps_tail}.pkl"))
    return hits[0] if hits else None


def resolve_model_path(path):
    """把 `--resume` 给的路径解析成 (模型 .zip, replay buffer .pkl 或 None)。

    传进来的按**不带后缀**写（`models/xxx/best/best_model`），写了 `.zip` 也认。
    模型 .zip 找不到时直接报错：返回 None 会让 `--resume` 静默退化成从头训练，
    而课程脚本里这是最贵的一种错。buffer 的命名见 `find_replay_buffer`。
    """
    p = pathlib.Path(path)
    if p.suffix in (".zip", ".pkl"):  # 后缀可省，写了也认
        p = pathlib.Path(str(p)[: -len(p.suffix)])
    model_path = pathlib.Path(str(p) + ".zip")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 不存在（--resume 给不带后缀的路径即可）")
    return model_path, find_replay_buffer(model_path)


def pad_replay_buffer(old_buffer_path, new_path, new_obs_dim: int) -> pathlib.Path:
    """把旧 replay buffer 的观测**补齐**到新维度（新增列填 0），而不是把它丢掉。

    **为什么不能简单清空（2026-10-13 实测，两次）**：R7 和 R12 两次"保留权重、清空 buffer"
    都塌了，R12 尤其干净：25k 步内 8/20 → 0/20，`最远 x` 0.32 m（狗原地不动），
    `train/actor_loss` 从 +4.6 一路冲到 **−126**，而同样条件但**没清 buffer** 的 R11
    在同一区间 actor_loss 是 +6 → +70（正常量级）。机理是 buffer 一空，critic 就只在
    一小撮"随机动作摔倒"的数据上学，Q 值尺度失控，actor 转身去追那个虚高的 Q。
    清空对一个**有 300 万步经验的 critic** 等于把它的样本分布整个换掉——
    这跟"从零开始"不是一回事：从零开始的 critic 本来就没有尺度可言。

    **补 0 的语义和权重侧是一致的**：新列的权重是 0，所以旧转移上"新增权重"的梯度
    恰好也是 0（`dL/dW_new = dL/dh × 0`）。于是新增的 16 维**只会从新采集的数据里学**，
    而旧的那部分网络继续用它 300 万步的解。代价是每条旧转移都被标成"前方是平地"，
    而近台阶开局占一半，这个标签确实是错的——但它只影响**共享列**对"扫描全是 0"
    这个输入的解读，而这一解读在权重侧本来就是恒等的（新列权重全 0），
    所以 critic 在旧分布上的行为**不比原来差**。

    只读原文件，结果写到 new_path（一个完整的 ReplayBuffer pickle，约 300 MB）。
    """
    import pickle

    from gymnasium import spaces
    from stable_baselines3.common.buffers import ReplayBuffer

    from go2_common.config import OBS_CLIP

    with open(old_buffer_path, "rb") as f:
        old = pickle.load(f)
    if not isinstance(old, ReplayBuffer):
        raise TypeError(f"{old_buffer_path} 里不是 ReplayBuffer，是 {type(old).__name__}")

    # 注意形状是 (buffer_size, n_envs, obs_dim) 而不是 (buffer_size, obs_dim)——
    # SB3 给每个并行环境各留一列。用 `...` 和 `shape[-1]` 写，两种布局都对。
    old_dim = int(old.observations.shape[-1])
    if old_dim == new_obs_dim:
        raise ValueError(f"{old_buffer_path} 的观测已经是 {new_obs_dim} 维，不需要补齐")
    if old_dim > new_obs_dim:
        raise ValueError(f"{old_buffer_path} 是 {old_dim} 维，比目标 {new_obs_dim} 维还大")

    # `old.buffer_size` 已经是**每个环境**的容量（SB3 在构造时做了 `buffer_size // n_envs`），
    # 所以这里要乘回去，否则再被除一次，缓冲区直接缩一倍。
    new = ReplayBuffer(
        old.buffer_size * old.n_envs,
        spaces.Box(-OBS_CLIP, OBS_CLIP, (new_obs_dim,), np.float32),
        old.action_space,
        device=old.device,
        n_envs=old.n_envs,
        handle_timeout_termination=getattr(old, "handle_timeout_termination", True),
    )
    n_new = new_obs_dim - old_dim
    for attr in ("observations", "next_observations"):
        dst = getattr(new, attr)
        dst[..., :old_dim] = getattr(old, attr)
        dst[..., old_dim:] = 0.0
    # 动作/奖励/终止位与观测维度无关，原样搬过来（含**超时标签**，
    # 丢了它 prepare_resume_buffer 会正确地拒绝加载）
    for attr in ("actions", "rewards", "dones", "timeouts"):
        getattr(new, attr)[:] = getattr(old, attr)
    new.pos = old.pos
    new.full = old.full

    new_path = pathlib.Path(new_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_path, "wb") as f:
        pickle.dump(new, f)
    return new_path


def checkpoint_obs_dim(model_path) -> int:
    """读 checkpoint 里**实际存着**的观测维度。

    不能拿当前 config 的 `obs_dim()` 当答案：那个说的是"现在想用几维"，
    而 checkpoint 是当初用什么维度训出来的。两者不等时才会触发暖启动。
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    src = pathlib.Path(model_path)
    src = pathlib.Path(str(src)[: -len(src.suffix)]) if src.suffix == ".zip" else src
    data, _, _ = load_from_zip_file(str(pathlib.Path(str(src) + ".zip")), load_data=True, device="cpu")
    return int(data["observation_space"].shape[0])


def warm_start_scan(model_path, out_path, new_obs_dim: int) -> pathlib.Path:
    """把**旧观测维度**的 checkpoint 扩成含地形扫描的新维度，新增输入列全部填 0。

    **为什么填 0 而不是随机初始化**：`Linear` 里新增列的权重为 0 时，
    `W_old @ x_old + 0 @ x_new + b` 与原来**逐位相同**，所以暖启动那一刻策略和旧模型
    一模一样（`test_training_fixes.py` 里有单测钉住这一点）。那 16 维扫描只能靠梯度
    自己长进来，不存在"随机初始化当场把练了 300 万步的策略打乱"的风险。
    这是让"加观测"这件事**可负担**的关键：不白扔旧模型。

    **critic 的列位置是个坑**：SB3 的 Q 网络吃的是 `[obs, action]` 拼接，
    新增的是 **obs 的列**，下标落在旧 obs 和动作之间，所以不能"在尾巴上接"：
    必须 `[:, :old]` 取旧 obs 列、`[:, old:new]` 填 0、`[:, new:]` 接旧的**动作**列。
    `critic_target` 必须同样处理——只改在线网会让两边一开始就叉开，白送一个偏差。

    optimizer 的一阶/二阶动量（Adam 的 exp_avg / exp_avg_sq）形状跟着权重走，
    不一起扩的话加载会直接报 shape 错；这里按**形状**匹配着一起扩，规则同上。

    只读原文件，把结果写到 out_path，返回 out_path。已经是新维度的会直接报错，
    而不是静默产生一个错位的模型。
    """
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

    from go2_common.config import OBS_CLIP

    # 入参按"不带后缀"写的也认（resolve_model_path 返回的就是不带 .zip 的），
    # 但**不能**把带后缀的原样喂给 load_from_zip_file：SB3 自己会再补一个 .zip。
    src = pathlib.Path(model_path)
    src = pathlib.Path(str(src)[: -len(src.suffix)]) if src.suffix == ".zip" else src
    src = pathlib.Path(str(src) + ".zip")

    data, params, pytorch_vars = load_from_zip_file(str(src), device="cpu")
    old_obs_dim = int(data["observation_space"].shape[0])
    if old_obs_dim == new_obs_dim:
        raise ValueError(f"{model_path} 的观测已经是 {new_obs_dim} 维，不需要暖启动")
    if old_obs_dim > new_obs_dim:
        raise ValueError(f"{model_path} 是 {old_obs_dim} 维，比目标 {new_obs_dim} 维还大，方向反了")
    n_new = new_obs_dim - old_obs_dim

    pol = params["policy"]
    action_dim = int(pol["actor.mu.weight"].shape[0])

    # --- actor：新列接在旧观测列后面 ---
    key = "actor.latent_pi.0.weight"
    w = pol[key]
    if w.shape[1] != old_obs_dim:
        raise ValueError(f"{key} 是 {tuple(w.shape)}，和 checkpoint 声明的 {old_obs_dim} 维对不上")
    zero = torch.zeros(w.shape[0], n_new, dtype=w.dtype)
    pol[key] = torch.cat([w, zero], dim=1)

    # --- critic（在线 + target）：观测列 | 新的 0 | 动作列 ---
    for name in ("critic.qf0.0.weight", "critic.qf1.0.weight",
                 "critic_target.qf0.0.weight", "critic_target.qf1.0.weight"):
        w = pol[name]
        if w.shape[1] != old_obs_dim + action_dim:
            raise ValueError(f"{name} 是 {tuple(w.shape)}，和 {old_obs_dim}+{action_dim} 对不上")
        pol[name] = torch.cat(
            [w[:, :old_obs_dim], torch.zeros(w.shape[0], n_new, dtype=w.dtype), w[:, old_obs_dim:]],
            dim=1,
        )

    # --- optimizer 动量：形状跟着权重，actor 和 critic 的规则不同 ---
    for opt_name, mode in (("actor.optimizer", "actor"), ("critic.optimizer", "critic")):
        state = params.get(opt_name, {}).get("state", {})
        for slot in state.values():
            for mom in ("exp_avg", "exp_avg_sq"):
                t = slot.get(mom)
                if t is None or t.ndim != 2:
                    continue
                if mode == "actor" and t.shape[1] == old_obs_dim:
                    slot[mom] = torch.cat([t, torch.zeros(t.shape[0], n_new, dtype=t.dtype)], dim=1)
                elif mode == "critic" and t.shape[1] == old_obs_dim + action_dim:
                    slot[mom] = torch.cat(
                        [t[:, :old_obs_dim], torch.zeros(t.shape[0], n_new, dtype=t.dtype), t[:, old_obs_dim:]],
                        dim=1,
                    )

    # --- 记录在 data 里的观测快照也要跟着改，否则 resume 第一次 predict 就 shape 炸 ---
    data["observation_space"] = spaces.Box(-OBS_CLIP, OBS_CLIP, (new_obs_dim,), np.float32)
    for k in ("_last_obs", "_last_original_obs"):
        v = data.get(k)
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == old_obs_dim:
            data[k] = np.concatenate([v, np.zeros((len(v), n_new), np.float32)], axis=1)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_zip_file(str(out_path), data, params, pytorch_vars, verbose=0)
    return out_path


def load_resume_configs(model_path: pathlib.Path):
    """从 checkpoint 同目录（或其上一层）的 config.json 还原 (EnvCfg, RewardCfg)。

    否则很容易出现"用 47 维的配置去加载 48 维的模型"这种事。
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
