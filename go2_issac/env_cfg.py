"""`CourseEnvCfg`：把 MuJoCo 那套环境接成 Isaac Lab 的 ManagerBasedRLEnv。

只做**接线**，不做任何语义：地形几何来自 `course.py`、观测/奖励/终止来自 `core.py`、
资产参数来自 `assets/go2_reference.json`。这个文件里的每个数字要么是"Isaac 侧的物理旋钮"
（PhysX 求解器、显存相关的规模），要么是从 `go2_common/config.py` 读出来的，
**没有第三个来源**——两边配置漂移是这次迁移最想避免的事。

三个和 MuJoCo 侧**故意不一样**的地方（都在下面标了 `[差异]`）：

1. `soft_joint_pos_limit_factor=1.0`：MuJoCo 侧是自己 clip 到 `limit ± joint_margin`，
   不需要资产再内缩一层。默认值 0.95 会让限位提前 5%，`q_default` 的 clip 结果就不一样了。
2. 地形的**物理材质摩擦 = 0.4**：MuJoCo 里地形 box 是 1.0、脚底球 0.4 且 `priority=1`
   （priority 高者独占该接触对），所以脚-地实际是 **0.4**。Isaac 没有 priority 这回事，
   两边都设 0.4，用任何合成方式（平均/最大/相乘）算出来都是 0.4。
3. `damping = 3.6`（不是 kd=3.5）：MuJoCo 的 12 个铰链自己带 `damping=0.1`，
   Isaac 的隐式执行器只有 `damping` 一个旋钮，两个要加起来。依据见
   `assets/go2_reference.json` 的 `joints[].damping`（实测 0.1）。
"""
from __future__ import annotations

import math
import pathlib

from go2_common.config import EnvCfg as Go2EnvCfg
from go2_common.config import RewardCfg
from go2_issac import course
from go2_issac.mdp import actions as course_actions
from go2_issac.mdp import commands, events, observations, rewards, state, terrain, terminations

import isaaclab.envs.mdp as base_mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    PhysxCfg,
    RigidBodyMaterialCfg,
    RigidBodyPropertiesCfg,
    SimulationCfg,
    UsdFileCfg,
)
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

HERE = pathlib.Path(__file__).resolve().parent
GO2_USD = str(HERE / "assets" / "go2.usd")

#: 参考环境配置（`go2_common/config.py`）。地形那两项在 `CourseEnvCfg.__post_init__` 里覆盖。
_REF = Go2EnvCfg()
#: 14 项奖励权重。**唯一来源**，两边共用（`mdp/rewards.py` 在运行期也从这里读）。
_REW = RewardCfg()

#: 地形几何参数来自 `course.py`（本机对着 MuJoCo 逐位对拍过）
_PATCH = course.PATCH_SIZE
_H_SCALE = course.HORIZONTAL_SCALE
_V_SCALE = course.VERTICAL_SCALE

TERRAIN_PRIM = state.TERRAIN_PRIM_PATH


def _patch_grid(num_envs: int) -> tuple[int, int]:
    """每个环境一块 5×5 m 的 patch，排成 `(行, 列)`。

    一个 patch 在内存里是 `(size/horizontal_scale)² = 250×250` 个 int16，整张地形
    是 `(行×250, 列×250)`。512 个环境 = 8×64 => 2000×16000 像素 ≈ 64 MB，
    第一次生成要几十秒，`use_cache=True` 之后从缓存读。**环境数翻 4 倍，地形内存翻 16 倍**
    （两个方向都变长），4096 个环境就是 512 MB——加环境数之前先算这一笔。
    """
    cols = 64
    rows = max(1, math.ceil(num_envs / cols))
    return rows, cols


@configclass
class CourseSceneCfg(InteractiveSceneCfg):
    """场景：狗 + 地形 + 接触传感器。

    `env_spacing` 在 `terrain_type="generator"` 下不生效（环境位置由 patch 决定），
    留着只是为了让 cfg 读起来完整。
    """

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        # MJCF 转出来的 USD 里带了**两个** `ArticulationRootAPI`（上机 dump 出来的）：
        #   /go2/base_link/base_link   躯干刚体——浮动基座的根就该是它
        #   /go2/worldBody             MJCF 那个空的 `<worldbody>` 残留
        # Isaac Lab 默认会在 `prim_path` 底下**找**那唯一的根，发现两个直接报错
        # （`articulation.py:1491` `Failed to find a single articulation ... Found multiple`）。
        # 这里把它指死，别去改 USD：`articulation_root_prim_path` 就是为这种情况留的，
        # 而且它同时决定了下面 `articulation_props`（solver 迭代、自碰撞）写在哪。
        #
        # 为什么是 base_link 而不是 worldBody：PhysX 的关节树只能顺着 prim 往下长，
        # 而 `worldBody` 是 `base_link` 的**兄弟**不是祖先，挂它底下一个刚体都找不到。
        # 官方口径也是这句（`schemas.py:85`）："For floating articulations, this should be
        # on the root body."——基座浮动，根必须在刚体上。
        #
        # 顺带：这层"多一个根"只在 `--check-asset` 的碰撞几何个数对不上时才会想歪，
        # 报错却是"找不到单个 articulation"，见到就往这里看。
        articulation_root_prim_path="/base_link/base_link",
        spawn=UsdFileCfg(
            usd_path=GO2_USD,
            # 不打开自碰撞：MuJoCo 侧也没开，开了之后大腿和机身会互顶，行为差很多
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 出生点由 `mdp/events.py:reset_base` 事件写，这里只是"资产默认态"，
            # 取 home 关键帧的高度（0.27）+ 一点余量
            pos=(0.0, 0.0, 0.30),
            joint_pos={name: float(v) for name, v in zip(state.JOINT_NAMES, [0.0, 0.9, -1.8] * 4)},
            joint_vel={".*": 0.0},
        ),
        # [差异 1] 1.0 = 用资产的硬限位；0.05 的余量由 `core.default_joint_pos` 自己留
        soft_joint_pos_limit_factor=1.0,
        actuators={
            # 对照 go2.xml 的 <motor> ctrlrange：髋/大腿 ±23.7、小腿 ±45.43
            "hip_thigh": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
                stiffness=_REF.kp,
                damping=_REF.kd + 0.1,   # [差异 3] 3.5 + 铰链自带的 0.1
                friction=0.2,            # 对照 dof_frictionloss
                effort_limit_sim=23.7,
            ),
            "calf": ImplicitActuatorCfg(
                joint_names_expr=[".*_calf_joint"],
                stiffness=_REF.kp,
                damping=_REF.kd + 0.1,
                friction=0.2,
                effort_limit_sim=45.43,
            ),
        },
    )

    #: 撞地形要扣分的那部分接触力（`reward_collision`）。脚底不算——脚本来就要碰地。
    #:
    #: `base_link/` 那一层**不能省**。Isaac 的路径通配是"一层一个 token"：
    #: `sim_utils.find_matching_prims` 把表达式按 `/` 切开，每个 token 编译成
    #: `^token$` 只跟**一个** prim 的名字比（`sim/utils.py:730-746`），
    #: `.*` 匹配的是"这一层里任意叫什么"的名字，**不跨 `/`**。
    #: 所以写 `Robot/.*` 只会命中 `Robot` 的 4 个直接子节点
    #: （`base_link`、`joints`、`Looks`、`worldBody`），一个刚体都没有，
    #: 接着 `contact_sensor.py:262` 报 "could not find any bodies with contact reporter API"。
    #: 官方 URDF 资产生效是因为那边刚体**就是** root 的直接子节点（`/anymal_c/LF_HIP`）；
    #: MJCF 转出来的多包了一层 `base_link/`（刚体在 `/Robot/base_link/<body>`），
    #: 照抄官方那行就正好差一层。实测（`/tmp/probe_contact.py`）加回这层后 17 个刚体全部命中。
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/.*",
        history_length=0,
        track_air_time=False,
        # 只要"和地形之间的"接触：MuJoCo 那边也是按 `geom_bodyid == 0` 筛的
        filter_prim_paths_expr=[TERRAIN_PRIM],
    )

    #: 地形 importer。**必须挂在 scene 上**：Isaac Lab 只扫 **scene** cfg 的成员找地形
    #: （`interactive_scene.py` 里那句 `isinstance(asset_cfg, TerrainImporterCfg)`），
    #: 挂到 env cfg 上是**静默失效**——不报错，但地形根本不生成、`env.scene.terrain is None`、
    #: `env_origins` 悄悄退回默认网格（于是整条课程坐标映射全错）。
    #: 内容由 `CourseEnvCfg.__post_init__` 填：patch 的行列数要按 `num_envs` 算，
    #: 写不成类属性默认值。
    terrain: TerrainImporterCfg | None = None


@configclass
class CourseActionsCfg:
    """12 个关节的位置增量。顺序按 `JOINT_NAMES`（= DDS 报文顺序）显式给。"""

    joints = course_actions.ClippedJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(state.JOINT_NAMES),
        # **`preserve_order` 必须开**：默认 False 时 `find_joints` 会把解析出来的下标
        # **按资产里的顺序排**，给进来的名字顺序被丢掉——实测动作项变成了
        # `FL, FR, RL, RR × hip/thigh/calf`（USD 导入顺序），而 `JOINT_NAMES` 是
        # `FR, FL, RR, RL × ...`（DDS 顺序）。动作写进关节目标这一路自己是自洽的
        # （`_joint_ids` 和 `_joint_names` 配套），但 `state.raw_action` 要按名字
        # 把它重排回 DDS 顺序才能和 `q/dq/tau`、和 MuJoCo 侧对齐——
        # 与其依赖那一次重排（多一处可能漏掉的地方），不如这里就让顺序一致。
        preserve_order=True,
        scale=_REF.action_scale,
        use_default_offset=True,
    )


@configclass
class CoursePolicyObsCfg(ObservationGroupCfg):
    """`policy` 这一个组，里面只有 1 个 term——**一个 term 装下 45 维**（理由见 `mdp/observations.py`）。

    Isaac Lab 的观测组是"**子类属性即 term**"：`ObservationManager` 遍历
    `group_cfg.__dict__` 收 term，再按名字跳过 `concatenate_terms`/`enable_corruption`
    这些组自己的字段。**没有 `terms=` 这个构造参数**——写错的话会在**类体执行**时
    （也就是 import 本模块的瞬间）就 `TypeError`，而不是等你建环境才报。
    """

    policy: ObservationTermCfg = ObservationTermCfg(func=observations.policy_obs)

    def __post_init__(self):
        self.concatenate_terms = True
        # 不做观测噪声：MuJoCo 侧没有，加了就不是同一个任务了
        self.enable_corruption = False


@configclass
class CourseObservationsCfg:
    """只有 `policy` 一组（`env_cfg` 里挂给 `ManagerBasedRLEnvCfg.observations`）。"""

    policy: CoursePolicyObsCfg = CoursePolicyObsCfg()


@configclass
class CourseRewardsCfg:
    """14 项，权重全部读 `RewardCfg`（`mdp/rewards.py` 里一项一行转调 `core.term_*`）。"""

    progress: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_progress, weight=_REW.progress)
    track_lin_vel: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_track_lin_vel, weight=_REW.track_lin_vel)
    climb: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_climb, weight=_REW.climb)
    level_bonus: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_level_bonus, weight=_REW.level_bonus)
    base_height: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_base_height, weight=_REW.base_height)
    lateral: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_lateral, weight=_REW.lateral)
    orientation: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_orientation, weight=_REW.orientation)
    yaw: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_yaw, weight=_REW.yaw)
    yaw_rate: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_yaw_rate, weight=_REW.yaw_rate)
    action_rate: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_action_rate, weight=_REW.action_rate)
    torques: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_torques, weight=_REW.torques)
    collision: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_collision, weight=_REW.collision)
    success: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_success, weight=_REW.success)
    fall: RewardTermCfg = RewardTermCfg(
        func=rewards.reward_fall, weight=_REW.fall)


@configclass
class CourseTerminationsCfg:
    """`time_out` 必须是这个名字且 `time_out=True`——rsl_rl 靠它填 `extras["time_outs"]`。"""

    time_out: TerminationTermCfg = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    success: TerminationTermCfg = TerminationTermCfg(func=terminations.success)
    fallen: TerminationTermCfg = TerminationTermCfg(func=terminations.fallen)
    flipped: TerminationTermCfg = TerminationTermCfg(func=terminations.flipped)
    out_of_course: TerminationTermCfg = TerminationTermCfg(func=terminations.out_of_course)


@configclass
class CourseCommandsCfg:
    """只有一个 vx 指令，只在重置时采样（理由见 `mdp/commands.py`）。"""

    course_velocity: commands.CourseVelocityCommandCfg = commands.CourseVelocityCommandCfg()


@configclass
class CourseEventCfg:
    """startup 建缓存、reset 写出生点。顺序就是执行顺序，别调换 `setup_state`。"""

    # startup：解析关节名 -> 建 Ctx（要等 articulation 实例化完）
    setup_state: EventTermCfg = EventTermCfg(func=state.setup, mode="startup")
    init_prev: EventTermCfg = EventTermCfg(func=observations.init_prev, mode="startup")
    # reset：出生点 + 关节初值（分布见 mdp/events.py）
    reset_base: EventTermCfg = EventTermCfg(func=events.reset_base, mode="reset")
    reset_joints: EventTermCfg = EventTermCfg(func=events.reset_joints, mode="reset")


@configclass
class CourseEnvCfg(ManagerBasedRLEnvCfg):
    """整个环境。`train.py` / `play.py` 建好它、改几个字段、注册成 gym 任务。"""

    #: flat / steps / full（见 `course.py`）
    terrain_variant: str = _REF.terrain
    #: 地形高度缩放，就是课程档位（0.3 最简单，1.0 原场景）
    terrain_scale: float = 1.0
    #: 动作幅度，单位 rad；决定狗能跨多高的槛（见 `EnvCfg.action_scale` 的实测表）
    action_scale: float = _REF.action_scale
    #: 观测里要不要第 46 维"机身高出地形的高度"（真机 DDS 拿不到，默认关）
    privileged: bool = _REF.privileged

    scene: CourseSceneCfg = CourseSceneCfg(num_envs=512, env_spacing=2.5)
    sim: SimulationCfg = SimulationCfg(
        dt=_REF.sim_dt,                       # 0.005 s，和 MuJoCo 侧一致
        render_interval=int(_REF.decimation),  # 4 -> 策略 50 Hz
        physx=PhysxCfg(
            solver_type=1,          # TGS：Isaac Lab 默认，接触稳定性和 MuJoCo 最接近的一档
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_found_lost_pairs_capacity=2**21,
            # MuJoCo 侧是 elliptic 锥 + impratio=100，PhysX 只有各向同性摩擦，
            # 这几个容量参数给足，免得环境一多就开始随机丢接触（表现为"训练不稳定"）
        ),
    )
    decimation: int = int(_REF.decimation)
    episode_length_s: float = _REF.max_episode_s   # 20 s -> 1000 个策略步

    actions: CourseActionsCfg = CourseActionsCfg()
    observations: CourseObservationsCfg = CourseObservationsCfg()
    rewards: CourseRewardsCfg = CourseRewardsCfg()
    terminations: CourseTerminationsCfg = CourseTerminationsCfg()
    commands: CourseCommandsCfg = CourseCommandsCfg()
    events: CourseEventCfg = CourseEventCfg()

    def __post_init__(self):
        """把地形 importer 挂上（网格要按 `scene.num_envs` 算，所以不能写成类属性默认值）。"""
        super().__post_init__()
        self._build_terrain()

    # -------------------------------------------------------------- 工具

    def _build_terrain(self) -> None:
        """建 `scene.terrain`。**挂 scene，不是挂 self**（理由见 `CourseSceneCfg.terrain`）。"""
        rows, cols = _patch_grid(self.scene.num_envs)
        self.scene.terrain = TerrainImporterCfg(
            prim_path=TERRAIN_PRIM,
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=_PATCH,
                border_width=0.0,
                num_rows=rows,
                num_cols=cols,
                horizontal_scale=_H_SCALE,
                vertical_scale=_V_SCALE,
                sub_terrains={
                    "course": terrain.CourseTerrainCfg(
                        variant=self.terrain_variant,
                        size=_PATCH,
                        horizontal_scale=_H_SCALE,
                        vertical_scale=_V_SCALE,
                    ),
                },
                # [重要] `difficulty` 就是 `terrain_scale`：区间收缩成一个点 = 固定档位。
                # 想要 "0.3 -> 1.0 自动加难" 就把 curriculum 打开、区间放宽（RUNBOOK 有说明）。
                difficulty_range=(self.terrain_scale, self.terrain_scale),
                curriculum=False,
                use_cache=True,
                seed=0,
            ),
            # [差异 2] 脚-地摩擦 0.4：MuJoCo 里脚底球 priority=1 独占接触对，
            # 所以地形 box 名义上的 1.0 其实用不上。两边都设 0.4 就没有"合成规则"这回事了。
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.4,
                dynamic_friction=0.4,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="average",
            ),
            debug_vis=False,
        )

    def set_envs(self, num_envs: int) -> None:
        """改环境数并重建地形网格（patch 数要 ≥ 环境数，否则多个环境共用一条跑道）。"""
        self.scene.num_envs = int(num_envs)
        self._build_terrain()

    def num_levels(self) -> int:
        """这个档位下有几级真台阶（`metrics` / `play.py` 打印"最高台阶 x/y"用）。"""
        return len(course.stair_tops_for(self.terrain_variant, self.terrain_scale))

    def to_go2_env_cfg(self) -> Go2EnvCfg:
        """折回 `go2_common.config.EnvCfg`，只为写 `config.json` / 还原。

        落盘用的必须是**这个**而不是 `CourseEnvCfg` 本身：那个里面有 USD 路径、
        PhysX 容量、`TerrainImporterCfg` 这些只对 Isaac 有意义的东西，写进 config.json
        既没法读也没法比。四个和 MuJoCo 侧同名的字段才是"这次跑的是哪个任务"的全部答案，
        和 `go2_ppo` 写出来的 JSON **逐字段同构**，两边可以直接 diff。
        """
        return Go2EnvCfg(
            terrain=self.terrain_variant,
            terrain_scale=self.terrain_scale,
            action_scale=self.action_scale,
            privileged=self.privileged,
        )

    def describe(self) -> str:
        """几行摘要，训练开始前打印（省得看 TensorBoard 才知道跑的是哪档）。"""
        ch = course.CourseHeight(self.terrain_variant, self.terrain_scale)
        tops = "  ".join(f"{t:.3f}" for t in ch.stair_tops) or "（没有台阶）"
        step_hz = 1.0 / (self.sim.dt * self.decimation)
        return (
            f"地形 {self.terrain_variant}@{self.terrain_scale:g}："
            f"方块 {len(ch.boxes)} 个，台阶顶面 {tops}\n"
            f"  {self.scene.num_envs} 个环境，{self.decimation} 子步 @ {self.sim.dt}s "
            f"-> 策略 {step_hz:.0f} Hz，回合 {self.episode_length_s:g}s = "
            f"{int(self.episode_length_s * step_hz)} 步\n"
            f"  action_scale {self.action_scale:g}，kp {_REF.kp:g} / "
            f"kd {_REF.kd:g}(+0.1 铰链阻尼)，观测 {'46' if self.privileged else '45'} 维"
        )


def build_env(cfg: CourseEnvCfg, render_mode: str | None = None):
    """建环境。**不注册 gym 任务**，直接实例化 `ManagerBasedRLEnv`。

    Isaac Lab 的示例脚本走的是 `gym.register` + `gym.make(id, cfg=cfg)`，但这里不需要
    那个字符串 id（没有 rl_games/外部工具按名字找任务），而注册这条路要多操心一件事：
    `spec.kwargs` 里的 `cfg` 和 `gym.make(cfg=...)` 谁覆盖谁、`disable_env_checker` 在
    哪个版本叫什么。`ManagerBasedRLEnv` 本身就是 `gym.Env`，直接 `ManagerBasedRLEnv(cfg=cfg)`
    得到的对象对 `RslRlVecEnvWrapper` 来说没有区别。**必须在 AppLauncher 起来之后调。**
    """
    from isaaclab.envs import ManagerBasedRLEnv

    return ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode)
