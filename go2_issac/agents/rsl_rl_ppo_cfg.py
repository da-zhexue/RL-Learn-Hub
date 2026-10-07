"""rsl_rl 的 PPO 超参——`go2_ppo/config.py:TrainCfg` 的逐项翻译。

**为什么换算法库**：rsl_rl 和 Isaac Lab 是一套（`RslRlVecEnvWrapper` / `OnPolicyRunner`
都是 Isaac Lab 自己的路径），全程 GPU、没有 Python 逐环境循环，吞吐比 SB3 高两个数量级。
代价是超参名全变了，所以下面把每一项的对应关系写死成表，改 `go2_ppo/config.py` 时
照着这张表同步（**不是**照抄数字，SB3 和 rsl_rl 的默认值/语义有差别的地方都标了）。

| go2_ppo/config.py | 这里 | 说明 |
|---|---|---|
| `learning_rate` 3e-4 | `algorithm.learning_rate` | 同 |
| `n_steps` 256 | `num_steps_per_env` | **不是** rsl_rl 默认的 24；256 步 = 5.12 s，理由见 TrainCfg 注释 |
| `n_epochs` 5 | `num_learning_epochs` | 同 |
| `batch_size` 256 | `num_mini_batches` | SB3 按**条数**分，rsl_rl 按**份数**分：`n_steps×n_envs/batch_size` |
| `gamma` 0.995 | `gamma` | 同 |
| `gae_lambda` 0.95 | `lam` | 同（名字不同！rsl_rl 叫 `lam`） |
| `clip_range` 0.2 | `clip_param` | 同 |
| `vf_coef` 0.5 | `value_loss_coef` | 同 |
| `max_grad_norm` 0.5 | `max_grad_norm` | 同 |
| `ent_coef` 0.005 | `entropy_coef` | 同 |
| `target_kl` 0.02 早停 | `schedule="adaptive"` + `desired_kl=0.02` | **正好是同一个东西**，不是近似 |
| `net_arch` (256,256) | `actor_hidden_dims` / `critic_hidden_dims` | 同 |
| `log_std_init` 0.0 | `init_noise_std` 1.0 | `log_std_init=0` ⇔ `σ=1` ⇔ `init_noise_std=1.0` |
| `total_steps` | `max_iterations` | `steps / (num_envs × num_steps_per_env)`，CLI 的 `--steps` 换算 |
| `clip_range_vf=None` | `use_clipped_value_loss=False` | 两边都是"不裁 value" |
| —— | `empirical_normalization=False` | MuJoCo 侧没有归一化；开了之后导出的策略还会带上归一化的统计量 |
"""
from __future__ import annotations

import math

from go2_ppo.config import TrainCfg

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

#: MuJoCo 侧的超参就是这个对象（不会有第二个默认值）
_SRC = TrainCfg()

#: 一个具体环境数下的 minibatch 数：`n_steps × n_envs / batch_size`（SB3 口径换算）
def mini_batches(n_envs: int) -> int:
    return max(1, int(round(_SRC.n_steps * n_envs / _SRC.batch_size)))


@configclass
class Go2IsaacPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """`train.py` 用它 + `num_steps_per_env` 建 runner。"""

    # rsl_rl 默认 24（= 0.48 s），对爬台阶来说太短——一个"助跑-抬腿-上台阶"的动作串
    # 要好几秒才兑现。用 go2_ppo 的值 256 步（5.12 s），回合长度（20 s）的四分之一。
    num_steps_per_env: int = _SRC.n_steps
    # 迭代上限由 CLI 的 `--steps` 换算，这里给个够大的默认值
    max_iterations: int = 1000
    # rsl_rl 默认会开一个经验归一化；MuJoCo 侧没有这个，开了之后策略的输入分布和
    # play.py 回放时不一致（同一份权重换个环境就崩），所以关掉
    empirical_normalization: bool = False
    # 存档策略组件：只存模型 + 观测归一化 + 状态字典，不存 optimizer（省一半体积）
    save_interval: int = 100
    experiment_name: str = "go2_isaac"
    logger: str = "tensorboard"
    # PPO 跑得快，默认 24 会刷屏
    log_interval: int = 1

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=math.exp(_SRC.log_std_init),   # log_std_init=0 -> σ=1.0
        actor_hidden_dims=list(_SRC.net_arch),
        critic_hidden_dims=list(_SRC.net_arch),
        activation="elu",       # rsl_rl 的默认，也是 legged_gym 一直用的那档
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=_SRC.learning_rate,
        num_learning_epochs=_SRC.n_epochs,
        num_mini_batches=mini_batches(_SRC.n_envs),
        gamma=_SRC.gamma,
        lam=_SRC.gae_lambda,
        clip_param=_SRC.clip_range,
        entropy_coef=_SRC.ent_coef,
        value_loss_coef=_SRC.vf_coef,
        # SB3 那边 `clip_range_vf=None` = 不裁 value；rsl_rl 的默认值是 True，所以要显式关掉
        use_clipped_value_loss=False,
        max_grad_norm=_SRC.max_grad_norm,
        # `target_kl=0.02`（SB3 是"超过 1.5×就停掉本轮剩下的 epoch"）⇔ rsl_rl 的
        # adaptive 调度（用实测 KL 反过来调学习率），两者的触发阈值是同一个数
        schedule="adaptive" if _SRC.target_kl is not None else "fixed",
        desired_kl=_SRC.target_kl if _SRC.target_kl is not None else 0.01,
    )


def iterations_for(total_steps: int, num_envs: int, num_steps_per_env: int | None = None) -> int:
    """`--steps`（总转移数）-> rsl_rl 的迭代数。

    一轮 = `num_envs × num_steps_per_env` 条转移。SB3 的"总步数"在向量环境下也是这个口径
    （`num_timesteps` 是各环境求和），所以 `--steps 3_000_000` 两边指的是同一件事。
    """
    per_iter = int(num_envs) * int(num_steps_per_env or _SRC.n_steps)
    return max(1, int(math.ceil(int(total_steps) / per_iter)))
