"""RL configuration for CartPole task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def cartpole_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for CartPole task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Actor：策略网络（随机策略），输出动作分布参数
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      # Critic：价值网络（确定性），输出 V(s)
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      # PPO 超参（裁剪、熵、学习率、KL 约束、梯度裁剪等）
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="selfcartpole",  # 实验名：用于日志与保存路径
    logger="tensorboard",  # 日志后端：TensorBoard
    save_interval=50,  # 每 50 次迭代保存一次 checkpoint
    num_steps_per_env=24,  # 每个环境每次 rollout 采样步数
    max_iterations=5_000,  # 最大训练迭代次数（可被 CLI 覆盖）
  )
