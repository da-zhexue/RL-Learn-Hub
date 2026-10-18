"""CartPole task registration."""

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import cartpole_env_cfg
from .rl_cfg import cartpole_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Cartpole",              # CLI 使用的任务名：train/play 后面跟这个字符串
  env_cfg=cartpole_env_cfg(),            # 训练配置：64 env、含随机 push 等
  play_env_cfg=cartpole_env_cfg(play=True),  # 播放配置：16 env、无 corruption、长 episode
  rl_cfg=cartpole_ppo_runner_cfg(),      # RL runner 配置：网络/算法/logger/max_iterations 等
  runner_cls=MjlabOnPolicyRunner,        # on-policy runner（负责采样、优化、保存等流程）
)