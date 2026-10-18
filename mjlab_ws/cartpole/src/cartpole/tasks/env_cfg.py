"""CartPole 任务环境配置"""

import copy
import math

import numpy as np
import torch

# 机器人实体配置
from cartpole.asset_zoo.robots.selfcartpole_constants import (
  get_selfcartpole_robot_cfg,
)

# Env 与常用 MDP 工具
from mjlab.envs import ManagerBasedRlEnvCfg, mdp

# 动作 项
from mjlab.envs.mdp.actions import JointVelocityActionCfg

# 事件 reset 工具
from mjlab.envs.mdp.events import reset_scene_to_default

# Manager 侧配置项
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

# 场景 仿真 地形 Viewer 配置
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


def _random_push_cart(env, env_ids, force_range=(-5, 5)):
  """训练时给小车随机推力 提升鲁棒性"""
  n = len(env_ids)
  # 在 force_range 内均匀采样推力
  random_forces = (
    torch.rand(n, device=env.device) * (force_range[1] - force_range[0])
    + force_range[0]
  )
  # qfrc_applied 是外加广义力 这里对第 0 个自由度即 slide 方向施加
  env.sim.data.qfrc_applied[env_ids, 0] = random_forces


# Upright ghost 命令 只做可视化 在 Viser 中画一个透明目标姿态
# 目标姿态定义为 cart 在 0 pole 竖直向上
class UprightGhostCommand(CommandTerm):
  """只用于 debug 可视化的 CommandTerm 不产生真实 command"""

  def __init__(self, cfg: "UprightGhostCommandCfg", env):
    super().__init__(cfg, env)
    self._ghost_model = None
    # ghost 颜色 RGBA 最后一位 alpha 在后面 add_ghost_mesh 里也会传入
    self._ghost_color = np.array((0.5, 0.7, 0.5, 0.5), dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    # 返回空 command 向量 表示该 term 不参与策略输入与控制
    return torch.zeros(self.num_envs, 0, device=self.device)

  def _update_metrics(self) -> None:
    # 这个 term 没有训练指标
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # 这个 term 不需要重采样
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # 这个 term 不需要更新 command
    del env_ids

  def _debug_vis_impl(self, visualizer) -> None:
    # visualizer 会根据当前 viewer 设置 返回需要绘制的 env 索引
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    if self._ghost_model is None:
      # 用当前仿真 mj_model 复制一个 ghost model
      # 然后把所有 geom_rgba 改成 ghost 颜色 这样渲染出来是统一半透明
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      self._ghost_model.geom_rgba[:] = self._ghost_color

    # upright 定义为 qpos 全零
    # CartPole 的 nq=2 依次是 slide 位移 与 hinge 角度
    qpos = np.zeros(self._env.sim.mj_model.nq)

    # mocap 用单位四元数 表示不旋转
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # env_origins 是每个子环境的世界坐标原点 用于多环境平铺
    env_origins = self._env.scene.env_origins.cpu().numpy()
    for batch in env_indices:
      # ghost 放到对应 env 的 origin 上
      mocap_pos = env_origins[batch]
      visualizer.add_ghost_mesh(
        qpos,
        model=self._ghost_model,
        mocap_pos=mocap_pos,
        mocap_quat=identity_quat,
        alpha=0.5,
        label=f"upright_ghost_{batch}",
      )


class UprightGhostCommandCfg(CommandTermCfg):
  """Upright ghost 的配置 仅用于可视化 不产生真实 command"""

  # resampling_time_range 是 CommandManager 的通用字段
  # 这里设成极大值 等价于不需要定期重采样
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  # debug_vis=True 才会调用 _debug_vis_impl 进行绘制
  debug_vis: bool = True

  def build(self, env) -> UprightGhostCommand:
    # 根据 cfg 构建对应 CommandTerm 实例
    return UprightGhostCommand(self, env)


def cartpole_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 CartPole 环境配置

  Args:
    play: 为 True 表示播放评估模式
      会关闭观测扰动 corruption
      会关闭 random_push 等训练扰动
      会把 episode_length 设得很大 方便长时间观看
  """
  # SceneCfg：定义多环境布局、地形与实体（机器人）
  scene_cfg = SceneCfg(
    num_envs=64 if not play else 16,
    extent=1.0,
    env_spacing=2.0,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_selfcartpole_robot_cfg()},
  )

  # ViewerConfig：相机跟随 robot 的 pole 物体，便于观察摆杆是否直立
  viewer_cfg = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="pole",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
  )

  # SimulationCfg：MuJoCo 仿真步长与求解迭代次数
  sim_cfg = SimulationCfg(
    mujoco=MujocoCfg(timestep=0.02, iterations=1),
  )

  # Actions：对 slide 关节的速度执行器施加控制（scale=20 对应 ctrlrange 量级）
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=20.0,
      use_default_offset=False,
    ),
  }

  # Observations：4 维（摆角/角速度/小车位置/小车速度），做简单归一化以稳定训练
  actor_terms = {
    "angle": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 1:2] / math.pi),
    "ang_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 1:2] / 5.0),
    "cart_pos": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 0:1] / 2.0),
    "cart_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 0:1] / 20.0),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      # 训练时可以加噪声扰动 提升鲁棒性
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      # critic 一般不加扰动 让价值估计更稳定
      enable_corruption=False,
    ),
  }

  def compute_upright_reward(env):
    """直立奖励 使用 cos(theta)"""
    # theta 越接近 0 cos(theta) 越接近 1
    return torch.cos(env.sim.data.qpos[:, 1])

  def compute_effort_penalty(env):
    """控制惩罚 抑制过大动作"""
    # ctrl 是执行器输入 对速度执行器来说近似是目标速度
    return -0.01 * (env.sim.data.ctrl[:, 0] ** 2)

  # Rewards：upright 用 cos(theta)（越直立越大）；effort 惩罚控制幅度
  rewards = {
    "upright": RewardTermCfg(func=compute_upright_reward, weight=5.0),
    "effort": RewardTermCfg(func=compute_effort_penalty, weight=1.0),
  }

  # Events：reset 时把机器人放回 env_origin，并给关节加一点随机扰动；训练时额外随机推车
  events = {
    "reset_scene_to_default": EventTermCfg(
      func=reset_scene_to_default,
      mode="reset",
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "position_range": (-0.1, 0.1),
        "velocity_range": (-0.1, 0.1),
      },
    ),
  }
  if not play:
    # 训练时加入随机推车 模拟外界扰动
    events["random_push"] = EventTermCfg(
      func=_random_push_cart,
      mode="interval",
      # 每 1 到 2 秒触发一次推车
      interval_range_s=(1.0, 2.0),
      # 推力范围更大 覆盖更多情况
      params={"force_range": (-20.0, 20.0)},
    )

  def check_pole_tipped(env):
    """摆杆是否倾倒 超过阈值则终止"""
    # 这里用 30 度作为倾倒阈值 你也可以根据任务难度调整
    return env.sim.data.qpos[:, 1].abs() > math.radians(30)

  # Terminations：超时或摆杆倾倒（>30°）终止
  terminations = {
    # timeout 是统一的时间终止项 由 episode_length_s 控制
    "timeout": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # tipped 为失败终止项 触发后结束本条轨迹
    "tipped": TerminationTermCfg(func=check_pole_tipped, time_out=False),
  }

  # Commands：upright_ghost 仅用于 Viser debug 可视化（透明目标姿态），不影响策略输入
  commands: dict[str, CommandTermCfg] = {
    "upright_ghost": UprightGhostCommandCfg(
      resampling_time_range=(1e9, 1e9),
      debug_vis=True,
    ),
  }

  return ManagerBasedRlEnvCfg(
    # scene 里包含 terrain 与机器人实体 也包含多环境布局信息
    scene=scene_cfg,
    # observations/actions/rewards/events/terminations/commands 分别由对应 manager 管理
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    events=events,
    terminations=terminations,
    # sim 定义 MuJoCo 仿真参数 timestep 等
    sim=sim_cfg,
    # viewer 定义相机默认行为 只影响可视化
    viewer=viewer_cfg,
    # decimation 表示每个 policy step 之间仿真步数
    # 这里为 1 表示每次 step 都做一次仿真推进
    decimation=1,
    # play 时 episode_length 设很大，便于长时间观看；训练时用 10s
    # episode_length_s 会影响 timeout 终止项
    episode_length_s=int(1e9) if play else 10.0,
  )
