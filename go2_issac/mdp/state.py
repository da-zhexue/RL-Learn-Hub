"""把 Isaac Lab 的 articulation 数据折成 `core.State`——观测/奖励/终止唯一的取数口。

**为什么不缓存**：`manager_based_rl_env.step()` 一步里会依次问终止、奖励、观测，
三次都得看到"同一时刻"的量。只要每次都从 `robot.data` 现读，这一点自动成立，
而缓存就得自己维护失效时机（重置、子步、版本差异），是个纯负债。代价是几个切片 + 一次
`terrain.top`（(N,8) 的比较），跟物理步比可以忽略。

**"上一拍"怎么办**：`core.State` 里 `prev_*` 四个字段（`action_rate`、三个势能项要用）
不能现读，得有人维护。维护点在 `mdp/observations.py`，那是**一步里最后跑的一件事**，
理由写在那个文件的 docstring 里。这里只负责"把 env 里存的那份填进 State"。

关节顺序：`JOINT_NAMES` 按 DDS 报文/执行器顺序（FR, FL, RR, RL × hip, thigh, calf）
**显式写死**，然后按名字去 articulation 里查下标。三个顺序（MJCF 的身体树、USD 的导入顺序、
DDS 报文）互不相同，只认名字不认下标——这是整个迁移里最容易静默错的一件事。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from go2_common.config import DEFAULT_Q, EnvCfg, RewardCfg
from go2_issac import core, course

F32 = torch.float32
F64 = torch.float64

# DDS/执行器顺序，`go2_common/config.py:27` 那句注释说的就是它
LEGS = ("FR", "FL", "RR", "RL")
PARTS = ("hip", "thigh", "calf")
JOINT_NAMES: tuple[str, ...] = tuple(f"{leg}_{part}_joint" for leg in LEGS for part in PARTS)

# 脚底球所在的 body（`go2.xml` 里 FL_foot 是 FL_calf 的定长孩子，只有一个球几何）。
# MJCF 转 USD 时这种零质量 body 有可能被并进父连杆，所以这里按名字查、查不到就退到小腿，
# 由 `smoke.py --check-asset` 打印实际解析到哪个。
FOOT_BODY_NAMES: tuple[str, ...] = tuple(f"{leg}_foot" for leg in LEGS)
CALF_BODY_NAMES: tuple[str, ...] = tuple(f"{leg}_calf" for leg in LEGS)

TERRAIN_PRIM_PATH = "/World/ground"


@dataclass
class Ctx:
    """每个 env 一份的缓存：名字解析结果、配置、地形查询器。

    这些都是"建好就不变"的东西（除了 `q_default`/`joint_limits` 也只在建的时候读一次），
    所以放这儿和 State 完全不同——State 每步都在变，Ctx 整个训练期间不动。
    """

    cfg: EnvCfg            # go2_common 的环境配置（含 action_scale / kp / kd / 阈值）
    reward_cfg: RewardCfg  # 14 项权重
    terrain: core.CourseTerrain
    course_height: course.CourseHeight
    joint_ids: list[int]   # 按 JOINT_NAMES 的顺序（= DDS 顺序）
    q_default: torch.Tensor       # (12,) float64，已按关节限位夹过
    joint_limits: torch.Tensor    # (12, 2) float64
    penalty_body_ids: list[int]   # ContactSensor 里"撞地形要罚"的 body 下标（不含脚）
    foot_body_ids: list[int]
    foot_body_names: tuple[str, ...]
    goal_z: float          # 成功判据的顶面高度（flat/steps 是 inf）
    cmd_term_name: str = "course_velocity"
    sens_name: str = "contact_forces"


def _ctx(env) -> Ctx:
    got = getattr(env, "_course_ctx", None)
    if got is None:
        raise RuntimeError(
            "CourseEnvCtx 还没建。`CourseEnvCfg` 会用 mode=\"startup\" 的事件调用 "
            "`mdp.state.setup(env)`，先确认 env_cfg 里的 events.setup_state 还在。")
    return got


def setup(env, env_ids=None) -> None:
    """建 `Ctx`（`mode="startup"` 事件，env 一建好就跑一次）。

    放在事件里而不是 `__post_init__`，是因为要等 scene/articulation 真的实例化出来
    （`robot.data.joint_pos_limits` 这些得 sim 起了才有值）。
    """
    cfg = EnvCfg(
        terrain=env.cfg.terrain_variant,
        terrain_scale=env.cfg.terrain_scale,
        action_scale=env.cfg.action_scale,
        privileged=env.cfg.privileged,
    )
    robot = env.scene["robot"]
    device = env.device

    joint_ids, joint_names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    if tuple(joint_names) != JOINT_NAMES:
        raise RuntimeError(
            "关节名字对不上（USD 里的名字和 go2.xml 不一致，或 preserve_order 没生效）：\n"
            f"  请求 {JOINT_NAMES}\n  实得 {tuple(joint_names)}")
    if len(joint_ids) != 12:
        raise RuntimeError(f"只找到 {len(joint_ids)} 个关节，应该是 12 个")

    limits = robot.data.joint_pos_limits[0][joint_ids].to(F64).clone()
    q_default = core.default_joint_pos(cfg, limits)

    terrain = core.CourseTerrain(cfg.terrain, cfg.terrain_scale, device=device, dtype=F64)
    course_height = course.CourseHeight(cfg.terrain, cfg.terrain_scale)

    # 接触传感器里的 body 顺序和 articulation 的 body 顺序不一定一样，按名字查
    foot_ids, foot_names = _resolve(robot.body_names, FOOT_BODY_NAMES, CALF_BODY_NAMES)
    sens = env.scene.sensors["contact_forces"]
    sens_bodies = list(sens.body_names)
    penalty_ids = [i for i, n in enumerate(sens_bodies) if n not in set(foot_names)]

    env._course_ctx = Ctx(
        cfg=cfg, reward_cfg=RewardCfg(), terrain=terrain, course_height=course_height,
        joint_ids=list(joint_ids), q_default=q_default, joint_limits=limits,
        penalty_body_ids=penalty_ids, foot_body_ids=foot_ids,
        foot_body_names=foot_names,
        goal_z=core.goal_z_for(cfg, course_height),
    )


def _resolve(body_names: list[str], wanted: tuple[str, ...],
             fallback: tuple[str, ...]) -> tuple[list[int], tuple[str, ...]]:
    """按名字找脚底 body；`FL_foot` 被并进 `FL_calf` 时退到小腿（理由见 FOOT_BODY_NAMES）。"""
    names: list[str] = []
    ids: list[int] = []
    for want, fb in zip(wanted, fallback):
        pick = want if want in body_names else fb
        if pick not in body_names:
            raise RuntimeError(f"body {want!r} 和退路 {fb!r} 都找不到；"
                               f"USD 里的 body 是 {body_names}")
        names.append(pick)
        ids.append(body_names.index(pick))
    return ids, tuple(names)


# ------------------------------------------------------------------ 取数


def course_xy(env, pos_w: torch.Tensor) -> torch.Tensor:
    """世界系 (x, y) -> 课程坐标 (x, y)。

    映射只有一份（`course.py` 的常量 + `world_to_course`），这里把它搬成 torch。
    `env.scene.env_origins` 是每个环境所属 patch 中心的世界坐标——
    **这个"中心"的说法是 Isaac Lab 的口径，上机第一个要确认的事**
    （`smoke.py --check-terrain` 用射线反推，对不上只改 `COURSE_ORIGIN_FROM_ENV_ORIGIN`）。
    """
    off = torch.tensor(course.COURSE_ORIGIN_FROM_ENV_ORIGIN, dtype=F64, device=pos_w.device)
    return pos_w[:, :2].to(F64) - env.scene.env_origins[:, :2].to(F64) - off


def terrain_h_and_level(ctx: Ctx, env, pos_w: torch.Tensor):
    """(当地地形高度, 站上第几级)，(N,) 各一。"""
    xy = course_xy(env, pos_w)
    return ctx.terrain.top(xy[:, 0], xy[:, 1]), ctx.terrain.level(xy[:, 0], xy[:, 1])


def raw_action(env) -> torch.Tensor:
    """动作（**夹到 ±1 之后**的值），(N,12)，顺序 = `JOINT_NAMES`。

    观测里的"上一拍动作"和 `action_rate` 用的就是这个值，两处都要求和 `env.py:194`
    一致——那边是先夹再存。夹取本身发生在 `mdp/actions.py:ClippedJointPositionAction`
    （只改关节目标），这里的夹是为了让 `prev_action` 也对。详见下面那段注释。

    **这里按名字把动作重排成 `JOINT_NAMES` 的顺序**，不依赖 `preserve_order`：
    动作项内部的关节顺序来自 `find_joints`（= USD 里的顺序），和 DDS 顺序不一定一样，
    而 `State.q/dq/tau` 用的是 `ctx.joint_ids`（DDS 顺序）。少了这一步，动作会和关节角
    **错位**——不报错，只是学不出来。`preserve_order=` 这个字段老版本没有，
    与其赌它存在，不如在这里按名字换一次位置（12 个元素，代价为零）。
    """
    term = env.action_manager.get_term("joints")
    got = getattr(term, "raw_actions", None)
    if not torch.is_tensor(got):                    # 老版本叫 action
        got = term.action
    # **必须自己夹到 ±1**：Isaac 的 `raw_actions` 是"网络吐出来的原值"，
    # 夹取发生在 `process_actions` 里（只改 processed_actions，不动 raw_actions）。
    # 而 `env.py:194` 是 `action = clip(action); self.prev_action = action` —— 那边
    # 存的**是夹过的值**。这不是边角情况：策略初始 σ=1.0，约 1/3 的动作 |a|>1，
    # 不夹的话观测里的"上一拍动作"和 MuJoCo 侧系统性对不上。
    got = torch.clamp(got.to(F64), -1.0, 1.0)
    names = list(term.joint_names)
    if names != list(JOINT_NAMES):
        perm = [names.index(n) for n in JOINT_NAMES]
        got = got[:, perm]
    return got


def get_state(env) -> core.State:
    """现读一份 `core.State`（无副作用，理由见模块 docstring）。"""
    ctx = _ctx(env)
    robot = env.scene["robot"]
    d = robot.data
    ids = ctx.joint_ids

    base_pos = d.root_pos_w.to(F64)
    terrain_h, terrain_level = terrain_h_and_level(ctx, env, base_pos)

    prev = getattr(env, "_course_prev", None)
    n = base_pos.shape[0]
    if prev is None:
        # startup 事件没跑成时的兜底：prev 全 0（等价于 env.py 里的初值）
        prev = {
            "prev_action": torch.zeros(n, 12, dtype=F64, device=base_pos.device),
            "prev_x": torch.zeros(n, dtype=F64, device=base_pos.device),
            "prev_terrain_h": torch.zeros(n, dtype=F64, device=base_pos.device),
            "prev_terrain_level": torch.zeros(n, dtype=torch.long, device=base_pos.device),
        }

    return core.State(
        base_pos=base_pos,
        base_quat=d.root_quat_w.to(F64),
        base_lin_vel=d.root_lin_vel_w.to(F64),
        base_ang_vel=d.root_ang_vel_b.to(F64),   # 陀螺：机体系
        q=d.joint_pos[:, ids].to(F64),
        dq=d.joint_vel[:, ids].to(F64),
        tau=d.applied_torque[:, ids].to(F64),
        contact_force=contact_force(env, ctx),
        action=raw_action(env),
        terrain_h=terrain_h,
        terrain_level=terrain_level,
        prev_x=prev["prev_x"],
        prev_action=prev["prev_action"],
        prev_terrain_h=prev["prev_terrain_h"],
        prev_terrain_level=prev["prev_terrain_level"],
        cmd_vx=cmd_vx(env, ctx),
    )


def cmd_vx(env, ctx: Ctx | None = None) -> torch.Tensor:
    """本回合的前进速度指令 (N,)，整回合不变（每回合只在重置时采一次）。"""
    ctx = ctx or _ctx(env)
    return env.command_manager.get_term(ctx.cmd_term_name).command[:, 0].to(F64)


def contact_force(env, ctx: Ctx | None = None) -> torch.Tensor:
    """非脚部件与地形之间的接触法向力之和 (N,)，牛顿。

    对着 `env.py:_contact_force()`：那边是把每个接触的**法向分量**取绝对值累加。
    Isaac 的 `ContactSensor` 给的是每个 body 的**合力**（世界系），所以这里取 z 分量、
    只保留推上去的正向部分再累加——平地/台阶上两者数值几乎一样，
    差别只在"斜着蹭台阶沿"这种接触上（那边法向不沿 z）。
    `RewardCfg.collision_force_free = 25 N` 的底噪本来就是为这种钝感留的。
    """
    ctx = ctx or _ctx(env)
    sens = env.scene.sensors[ctx.sens_name]
    f = sens.data.net_forces_w.to(F64)          # (N, B, 3) 世界系
    if f.dim() == 4:                            # 有的版本带 history 维
        f = f[:, -1]
    pick = f[:, ctx.penalty_body_ids, 2]
    return torch.clamp(pick, min=0.0).sum(dim=-1)
