"""go2_ppo 专属的训练超参。

常量、EnvCfg / RewardCfg、四元数工具都在 `go2_common/config.py`（SAC / PPO 共用）；
这里只有 PPO 自己的 `TrainCfg`。和 go2_sac 的字段名刻意保持同名，
凡是两边都有的（`gamma` / `eval_freq` / `net_arch` …）语义也一致。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainCfg:
    """PPO 超参。环境/SIM 时序和 SAC 那套完全一样（同一个 env，50 Hz）。"""

    total_steps: int = 3_000_000
    n_envs: int = 8
    seed: int = 0

    # --- PPO
    learning_rate: float = 3e-4
    # **256，不是 SB3 默认的 2048。** 策略 50 Hz、回合上限 20 s = 1000 步，
    # 2048 步是 41 s —— 比整个回合还长，一轮 rollout 里同一个回合的陈旧数据反复出现，
    # 优势估计基本失效。256 步 = 每个环境采 5.12 s，是回合长的四分之一到一半。
    n_steps: int = 256
    # 256 × 8 = 2048 条样本 -> 8 个 minibatch。同时它对**任意** n_envs 都整数：
    # 256·k ÷ 256 = k，换 --n-envs 不会冒出 SB3 那个"最后一个不满的 minibatch"警告。
    batch_size: int = 256
    # 一轮样本做 5 × 8 = 40 次梯度更新，与 legged_gym 同量级。
    n_epochs: int = 5
    # 视野约 4 s，和 SAC 一致。0.99 只有 2 s，规划不了"助跑-抬腿-上台阶"这一串动作。
    gamma: float = 0.995
    gae_lambda: float = 0.95  # GAE 的标准值
    # SB3 默认值。不裁 value（clip_range_vf=None）：这套奖励不归一化、尺度大（走完全程约 190 分），
    # 裁 value 等于按初始 value 的量级去限制它后面能长到哪。
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    # 熵奖励。PPO 的初始 σ=1.0（= legged_gym 的 init_noise_std），本来就远大于
    # SAC 崩掉后的 σ≈0.25，所以熵不是"救探索"的主旋钮；给一点是防过早收敛到局部最优。
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # 早停阀：某次更新里 approx_kl > 1.5 × 0.02 就停掉本轮剩下的 epoch。
    # 这个仓库的历史就是"微调把策略训坏"，多一道保险。None = 关闭。
    target_kl: float | None = 0.02
    net_arch: tuple = (256, 256)
    log_std_init: float = 0.0  # σ=1.0

    # --- 运行
    # 子进程各占 1 线程跑物理，父进程这一项给 torch 的梯度更新用（同 go2_sac）。
    torch_threads: int = 8
    eval_freq: int = 50_000
    # **20 而不是 5。** 5 个回合的成功率标准误是 ±22 个百分点，而 best_model 是在所有
    # 评估里取极大值——那是在按噪声选，选出来的是幸运样本。实测同一个模型：
    # 5 个回合报 4/5（80%），40 个回合只有 12/40（30%）。
    eval_episodes: int = 20
    checkpoint_freq: int = 200_000
