"""Go2 过地形强化学习环境（Gymnasium）。

**进程内直接跑 MuJoCo，不走 DDS**：DDS 仿真被实时锁在 200 Hz 且只有单环境。
训练地形就是 unitree_mujoco 自带的那份 scene.xml。

观测是 45 维、全部能从 DDS 的 LowState 复现，所以 `play.py --mode dds`
不用改任何东西就能把同一套策略发到仿真或真机上。
"""

from __future__ import annotations

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium import Env as GymEnv

from . import terrain
from .config import (
    DEFAULT_Q,
    OBS_CLIP,
    OBS_CMD_SCALE,
    OBS_DQ_SCALE,
    OBS_GYRO_SCALE,
    EnvCfg,
    RewardCfg,
    obs_dim,
    projected_gravity,
    quat_to_rpy,
)
from .reward import RewardInput, compute_reward

FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")


class Go2TerrainEnv(GymEnv):
    """Go2 在 scene.xml 地形上行走/爬台阶。

    动作 12 维（每关节一个位置增量，范围 -1~1），策略 50 Hz，
    底层按 kp/kd 算力矩、500/200 Hz 积分 —— 与 unitree_mujoco 的 DDS 桥完全一致。
    """

    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, cfg: EnvCfg | None = None, reward_cfg: RewardCfg | None = None):
        super().__init__()
        self.cfg = cfg or EnvCfg()
        self.reward_cfg = reward_cfg or RewardCfg()

        self.model = terrain.build_model(self.cfg.scene, self.cfg.terrain, self.cfg.terrain_scale)
        self.model.opt.timestep = self.cfg.sim_dt
        self.data = mujoco.MjData(self.model)
        self.terrain = terrain.TerrainHeight(self.model)
        # 成功判据里的"站上顶平台"高度按地形量出来。没有台阶的课程（flat / steps）
        # 就没有"顶平台"这回事，高度设成够不着（理由见 AGENT.md §5-C4）。
        self.cfg.goal_z = (
            self.terrain.top(self.cfg.goal_x, 0.0) + self.cfg.goal_clearance
            if self.terrain.stair_tops
            else np.inf
        )
        self.max_steps = int(round(self.cfg.max_episode_s / (self.cfg.sim_dt * self.cfg.decimation)))

        self._build_index_maps()
        self._build_contact_masks()

        n = self.model.nu
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = spaces.Box(
            -OBS_CLIP, OBS_CLIP, (obs_dim(self.cfg.privileged),), np.float32
        )

        self.cmd_vx = self.cfg.cmd_vx[0]
        self.prev_action = np.zeros(n, dtype=np.float64)
        # prev_x 必须也有初值：diag.hold_stance 会绕过 reset() 直接 step()，
        # 少了它 progress 项在第一次 step 就 AttributeError。
        self.prev_x = 0.0
        self.prev_terrain_h = 0.0
        self.prev_terrain_level = 0
        self.step_count = 0

    # -------------------------------------------------------------- 初始化

    def _build_index_maps(self):
        """执行器顺序 <-> qpos/qvel 地址。

        qpos 的关节顺序跟着**身体树**走（FL, FR, RL, RR），而执行器、DDS 报文、
        go2.xml 里的 sensor 全是 FR, FL, RR, RL，两边不一样。
        """
        jnt = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[jnt]
        self.vadr = self.model.jnt_dofadr[jnt]

        # 关节限位/力矩限幅，直接用模型里的值，别再抄一份常量
        margin = self.cfg.joint_margin
        self.q_lo = self.model.jnt_range[jnt, 0] + margin
        self.q_hi = self.model.jnt_range[jnt, 1] - margin
        self.tau_lo = self.model.actuator_ctrlrange[:, 0]
        self.tau_hi = self.model.actuator_ctrlrange[:, 1]
        self.q_default = np.clip(np.array(DEFAULT_Q, dtype=np.float64), self.q_lo, self.q_hi)

        self.ngeom = self.model.ngeom

    def _build_contact_masks(self):
        """预计算"哪些几何体算撞地形"的布尔掩码，避免每步查名字。"""
        self.is_terrain_geom = self.model.geom_bodyid == 0

        foot_ids = set()
        for name in FOOT_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise RuntimeError(f"go2.xml 里找不到脚部几何体 {name!r}")
            foot_ids.add(gid)
        ngeom = self.model.ngeom
        self.is_penalty_geom = np.array(
            [self.model.geom_bodyid[g] != 0 and g not in foot_ids for g in range(ngeom)]
        )
        # 顺序无所谓（support_height 取中位数），但固定下来便于调试
        self.foot_geom_ids = np.array(sorted(foot_ids), dtype=int)

    def _support_height(self) -> float:
        """狗实际站立的地面高度（四只脚下方的中位数），见 `terrain.support_height`。

        这是 `step()` 里唯一的"当地地形高度"来源：`climb` / `level_bonus` /
        `base_height` / `is_fallen` 全都吃它。换成机身中心处的 `terrain.top()`
        会让狗在没爬上去的时候就领到整级台阶的分（实测证据见 terrain.py 的注释）。
        """
        xy = self.data.geom_xpos[self.foot_geom_ids][:, :2]
        return self.terrain.support_height(xy)

    def support_level(self) -> int:
        """当前站上第几级——按**支撑面**算，和 `step()` 里发 `level_bonus` 的那个数同源。

        诊断/回放看"最高台阶"要用这个，别用 `terrain.level(x, y)`：后者是**机身中心**
        的口径，机身探到台阶上方时它就已经算上去了，会比狗真正站上的级数高一级
        （这正是旧版奖励虚高的那件事，见 `terrain.support_height`）。
        """
        return self.terrain.level_at_height(self._support_height())

    # -------------------------------------------------------------- 观测

    def _obs(self) -> np.ndarray:
        d = self.data
        quat = d.qpos[3:7]
        # d.qvel[3:6] 是**机体系**角速度（自由关节的线速度才是世界系），可以直接当陀螺用
        obs = np.concatenate(
            [
                d.qvel[3:6] * OBS_GYRO_SCALE,
                projected_gravity(quat),
                np.array([self.cmd_vx, 0.0, 0.0]) * OBS_CMD_SCALE,
                d.qpos[self.qadr] - self.q_default,
                d.qvel[self.vadr] * OBS_DQ_SCALE,
                self.prev_action,
            ]
        )
        if self.cfg.privileged:
            # 只有这一维在 DDS 的 LowState 里拿不到（要用 rt/sportmodestate），默认关。
            # 基准用支撑面（四只脚下方），和奖励/终止同一套口径
            obs = np.append(obs, d.qpos[2] - self._support_height())
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    # -------------------------------------------------------------- 接触

    def _contact_force(self) -> float:
        """非脚部件与地形之间的接触法向力之和（牛顿）。"""
        total = 0.0
        res = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            hit = (self.is_terrain_geom[g1] and self.is_penalty_geom[g2]) or (
                self.is_terrain_geom[g2] and self.is_penalty_geom[g1]
            )
            if not hit:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, res)
            total += abs(float(res[0]))
        return total

    # -------------------------------------------------------------- reset / step

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self.cfg
        rng = self.np_random

        x = float(rng.uniform(*cfg.reset_x))
        y = float(rng.uniform(-cfg.reset_y, cfg.reset_y))
        yaw = float(rng.uniform(-cfg.reset_yaw, cfg.reset_yaw))
        self.cmd_vx = float(rng.uniform(*cfg.cmd_vx))

        d = self.data
        mujoco.mj_resetDataKeyframe(self.model, d, self._home_key_id)
        d.qpos[0] = x
        d.qpos[1] = y
        # home 关键帧的脚底本来就陷进地面约 18 mm，这里抬到地形上方让它自己落稳。
        # **按整只狗的脚印**取地形高度，不能只用中心点（理由见 AGENT.md §5-C1）。
        d.qpos[2] = self.terrain.top_footprint(x, y) + float(rng.uniform(*cfg.reset_z))
        d.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        d.qvel[:] = 0.0

        self.prev_action = np.zeros(self.model.nu, dtype=np.float64)
        self.prev_x = x
        self.step_count = 0
        # 前推一次，否则接触/传感器数据还是上一回合的
        mujoco.mj_forward(self.model, d)
        # 支撑面必须在 mj_forward **之后**取：在那之前 geom_xpos 还停在上一回合的位置，
        # 四只脚读到的是上一次落点（出生点离台阶近时差一整级台阶）。
        self.prev_terrain_h = self._support_height()
        self.prev_terrain_level = self.terrain.level_at_height(self.prev_terrain_h)

        info = {
            "x": x,
            "y": y,
            "base_z": float(d.qpos[2]),
            "terrain_h": self.prev_terrain_h,
            "cmd_vx": self.cmd_vx,
        }
        return self._obs(), info

    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # 位置目标 + 每步力矩 PD，与 unitree_sdk2py_bridge.LowCmdHandler 的算法一致
        q_target = np.clip(self.q_default + cfg.action_scale * action, self.q_lo, self.q_hi)
        for _ in range(cfg.decimation):
            q = self.data.qpos[self.qadr]
            dq = self.data.qvel[self.vadr]
            self.data.ctrl[:] = np.clip(
                cfg.kp * (q_target - q) - cfg.kd * dq, self.tau_lo, self.tau_hi
            )
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        d = self.data
        base_pos = d.qpos[:3]
        rpy = quat_to_rpy(d.qpos[3:7])
        # 支撑面高度 = 四只脚下方的中位数，**不是**机身中心处的 top()（见 _support_height）
        terrain_h = self._support_height()
        terrain_level = self.terrain.level_at_height(terrain_h)

        success = terrain.is_success(cfg, base_pos)
        terminated = bool(
            success
            or terrain.is_fallen(cfg, terrain_h, base_pos)
            or terrain.is_flipped(cfg, rpy)
            or terrain.is_out_of_course(cfg, base_pos)
        )
        # 超时走 truncated 而不是 terminated，SB3 才会正确地做 bootstrap
        truncated = bool((not terminated) and self.step_count >= self.max_steps)

        r = RewardInput(
            action=action,
            prev_action=self.prev_action,
            q=d.qpos[self.qadr].copy(),
            dq=d.qvel[self.vadr].copy(),
            tau=d.actuator_force.copy(),
            quat=d.qpos[3:7].copy(),
            base_v=d.qvel[0:3].copy(),
            base_w=d.qvel[3:6].copy(),
            base_pos=base_pos.copy(),
            prev_x=self.prev_x,
            terrain_h=terrain_h,
            prev_terrain_h=self.prev_terrain_h,
            terrain_level=terrain_level,
            prev_terrain_level=self.prev_terrain_level,
            cmd_vx=self.cmd_vx,
            contact_force=self._contact_force(),
            terminated=terminated,
            success=success,
        )
        reward, parts = compute_reward(r, self.reward_cfg)

        self.prev_action = action
        self.prev_x = float(base_pos[0])
        self.prev_terrain_h = terrain_h
        self.prev_terrain_level = terrain_level

        info = {
            **parts,
            "x": float(base_pos[0]),
            "y": float(base_pos[1]),
            "base_z": float(base_pos[2]),
            "terrain_h": terrain_h,
            "vx": float(r.base_v[0]),
            "cmd_vx": self.cmd_vx,
            "level": terrain_level,
            "success": success,
            "is_success": success,  # SB3 的 Monitor 会用它统计
            # reward 分项里的 "success"（权重 20.0）被上面这个布尔值覆盖了，改名另存一份
            "success_bonus": parts["success"],
        }
        return self._obs(), float(reward), terminated, truncated, info

    # -------------------------------------------------------------- 杂项

    @property
    def _home_key_id(self) -> int:
        """home 关键帧下标（缓存一次）。"""
        if not hasattr(self, "_home_key"):
            self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if self._home_key < 0:
                raise RuntimeError("go2.xml 里没有名为 home 的关键帧")
        return self._home_key


def run_episode(env: Go2TerrainEnv, act, seed: int | None = None, on_step=None):
    """跑一个完整回合，返回 (累计奖励, 结束时的 info)。

    act(obs, env) -> action；on_step(env) 用来渲染或打印，可以为 None。
    train.py 的评估回调和 play.py 都用它，免得各写一份回合循环。
    """
    obs, info = env.reset(seed=seed)
    total = 0.0
    while True:
        action = act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if on_step is not None:
            on_step(env)
        if terminated or truncated:
            break
    return total, info
