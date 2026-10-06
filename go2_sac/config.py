"""go2_sac 专属的训练超参。

常量、EnvCfg / RewardCfg、四元数工具都在 `go2_common/config.py`（SAC / PPO 共用）；
这里只有 SAC 自己的 `TrainCfg`。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainCfg:
    """训练超参。默认值针对"CPU、无 GPU、n_envs=8 时约 440 环境步/秒"调的。"""

    total_steps: int = 3_000_000
    n_envs: int = 8
    seed: int = 0

    # --- SAC
    learning_rate: float = 3e-4
    buffer_size: int = 500_000  # 约 230 MB
    # 默认的 100 步在 50 Hz 下等于还没站起来就开始学，必须抬高
    learning_starts: int = 10_000
    batch_size: int = 256
    tau: float = 0.005
    # 视野约 4 s。0.99 只有 2 s，规划不了"助跑-抬腿-上台阶"这一串动作
    gamma: float = 0.995
    train_freq: int = 1
    # **必须是 -1，不能是 1。** 这个数不是"每次更新几步"，而是"每收集一轮做几次更新"，
    # 而一轮 = train_freq × n_envs = 1 × 8 = 8 个环境步（collect_rollouts 里
    # `self.num_timesteps += env.num_envs`）。写 1 就等于**每个梯度更新配 8 个环境步**，
    # 更新/数据比只有 1/8。实测旧模型 checkpoint：_n_updates=412,498 / num_timesteps=3,300,024
    # = **0.1250**，330 万环境步只做了 41 万次更新——整个课程一直只跑着应有意量的八分之一。
    # -1 是 SB3 的"收集多少步就更新多少次"（gradient_steps = rollout.episode_timesteps = 8），
    # 正好给出标准的 UTD = 1。代价是每环境步的墙钟时间约 8 倍。
    gradient_steps: int = -1
    net_arch: tuple = (256, 256)
    # 探索噪声。**这两个值决定了这套训练能不能跑起来**，别按常规的 -1.0 去调：
    # 实测 -1.0（σ=0.37）时，SAC 的自动熵系数会迅速衰减（α→0.024），200k 步后 σ 只剩 0.25，
    # 折算到关节上只有 ±0.06 rad 的抖动；而能让狗真正走起来的开环小跑需要 ±0.25 rad 的相干振荡。
    # 结果 buffer 里全是"站着"的数据，Q 函数只学到"站着最好"，actor 永远不迈腿
    # （200k 步平均 vx 只有指令的 0.00 倍，脚最多抬 7 cm，上不了 8 cm 的槛）。
    # 注意这和奖励无关：同一个 env 里手写一个 ±0.25 rad、3 Hz 的对角小跑，10 秒能走 4.57 m。
    log_std_init: float = 0.0
    # SB3 默认 target_entropy = -动作维度 = -12，对 12 维动作空间过强，会一直把 α 往下压。
    # 取一半（-6），让探索噪声维持住。
    target_entropy: float = -6.0

    # --- 演示数据预热（原理见 train.seed_replay_buffer 的注释）
    # 用开环小跑采一批数据填进 replay buffer，让 Q 一开始就知道"走路比站着强"。
    # 0 = 关闭。关闭后 SAC 会卡在"站着不动"的局部最优（实测 200k 步平均 vx 仍是 0.00×指令）。
    seed_trot_episodes: int = 30
    seed_rand_episodes: int = 6  # 另外混一些站立/随机动作，让 Q 能对比出高下

    # --- 运行
    # 子进程各占 1 线程跑物理，父进程这一项给 torch 的梯度更新用。
    # 实测：单独测一次 SAC 更新，4 线程 17 ms、8 线程 4 ms；实际训练循环里 4/8 都是
    # 约 610 步/秒（n_envs=8），差别不大。默认给 8，反正不亏。
    torch_threads: int = 8
    eval_freq: int = 50_000
    # **20 而不是 5。** 5 个回合的成功率标准误是 ±22 个百分点，而 best_model 是在所有
    # 评估里取极大值——那是在按噪声选，选出来的是幸运样本。实测同一个模型：
    # 5 个回合报 4/5（80%），40 个回合只有 12/40（30%）。这就是"每一级训练都把模型练坏了"
    # 这个结论的来源：拿一个幸运样本的旧评估去比一个新评估，必然下降。
    eval_episodes: int = 20
    checkpoint_freq: int = 200_000
