"""重置：出生点、关节初值。分布与 `env.py:reset()` 逐项一致。

三件事凑齐才算"和 MuJoCo 从同一个地方开始"：

1. **x/y/yaw**：`x ~ U(-0.3, 0.3)`、`y ~ U(-0.15, 0.15)`、`yaw ~ U(-0.20, 0.20)`（弧度）。
   yaw 是绕 z 的，四元数 (cos(yaw/2), 0, 0, sin(yaw/2))。
2. **z = `top_footprint`(x,y) + U(0.30, 0.33)**。**必须按整只狗的脚印取最高地形**，
   不能只用中心点：狗横跨台阶立面时中心还在平地、前脚已经探到台阶上方，
   用中心点的高度会让狗直接生在地里（见 AGENT.md §5-C1）。home 关键帧的脚底本来就陷进
   地面约 18 mm，那 0.30~0.33 m 的抬升就是让它自己落稳用的。
3. **关节初值 = `q_default`（hip 0 / thigh 0.9 / calf -1.8），关节速度 0**。
   MuJoCo 那边是 `mj_resetDataKeyframe(home)`，home 关键帧的关节值就是这个。

**坐标系**：`sample_reset` 给的是**课程坐标**，写进仿真前要换成世界系——
课程 (0,0) 在 patch 中心 + `COURSE_ORIGIN_FROM_ENV_ORIGIN` 处（映射只有 `course.py` 一份）。
z 不用换：地形 patch 坐落在世界 z=0 上，course 的"地形高度"就是世界高度。
"""
from __future__ import annotations

import torch

from go2_issac import core, course
from go2_issac.mdp import state

F64 = torch.float64


def reset_base(env, env_ids=None, **kwargs) -> None:
    """按 `env.py:160-173` 的分布写机身位姿（位置 + 偏航 + 线/角速度清零）。"""
    if env_ids is None:
        return
    ctx = state._ctx(env)
    device = env.device
    n = len(env_ids)

    gen = _rng(env)
    sample = core.sample_reset(ctx.cfg, ctx.terrain, n, generator=gen)

    origins = env.scene.env_origins[env_ids].to(F64)          # (n,3) patch 中心
    off = torch.tensor(course.COURSE_ORIGIN_FROM_ENV_ORIGIN, dtype=F64, device=device)
    pos = torch.zeros(n, 3, dtype=F64, device=device)
    pos[:, 0] = origins[:, 0] + off[0] + sample["x"]
    pos[:, 1] = origins[:, 1] + off[1] + sample["y"]
    pos[:, 2] = sample["z"]

    root_pose = torch.cat([pos, sample["quat"]], dim=-1).to(torch.float32)
    root_vel = torch.zeros(n, 6, dtype=torch.float32, device=device)
    robot = env.scene["robot"]
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def reset_joints(env, env_ids=None, **kwargs) -> None:
    """关节 = `q_default`、关节速度 = 0（home 关键帧那一套）。"""
    if env_ids is None:
        return
    ctx = state._ctx(env)
    robot = env.scene["robot"]
    n = len(env_ids)
    q = ctx.q_default.to(torch.float32).expand(n, -1).clone()
    dq = torch.zeros(n, 12, dtype=torch.float32, device=env.device)
    robot.write_joint_state_to_sim(q, dq, joint_ids=ctx.joint_ids, env_ids=env_ids)


def _rng(env) -> torch.Generator:
    """采样用的随机源（和网络权重/环境随机化各用各的，互不干扰）。

    种子取自 `env.cfg.seed`（Isaac Lab 的 `seed()` 已经把它设成了 `--seed`）。
    这样同一个 `--seed` 两次训练出生点序列一致，便于对比改动的效果。
    """
    got = getattr(env, "_course_rng", None)
    if got is None:
        got = torch.Generator(device=env.device)
        got.manual_seed(int(getattr(env.cfg, "seed", 0) or 0))
        env._course_rng = got
    return got
