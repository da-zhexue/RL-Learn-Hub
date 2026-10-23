"""Go2 的域随机化（domain randomization）事件。

mjlab 的 DR 统一走 events：每个 EventTermCfg 在启动（startup）或回合重置
（reset）时调用 `mjlab.envs.mdp.dr.*` 改写 MuJoCo 模型参数。这里集中放 Go2
的随机化项和范围，由 `Go2EnvCfg.dr_scale` 控制整体幅度：

* ``dr_scale = 0``：不注册任何 DR 项，模型与 XML 完全一致；
* ``dr_scale = 1``：使用下面写出的完整范围；
* 中间值：每个范围朝自己的中性点（模型默认值 / 0 / 1）线性收缩。

调范围改这个文件；训练时用 ``--env.dr-scale 0.5`` 这种写法调幅度。
"""

from __future__ import annotations

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .context import FOOT_GEOM_NAMES

# 中性点：dr_scale=0 时范围收缩到这些值，等价于关闭该项。
# 足底 geom 在 go2.xml 里的默认摩擦是 "0.4 0.02 0.01"。
_FOOT_FRICTION_DEFAULT = 0.4
# scale 类型（乘在模型默认值上）的中性点。
_SCALE_NEUTRAL = 1.0
# 加性偏移（质心、编码器零位、推力）的中性点。
_ZERO_NEUTRAL = 0.0


def _shrink(
    rng: tuple[float, float], neutral: float, scale: float
) -> tuple[float, float]:
    """把范围朝中性点线性收缩：scale=1 原样返回，scale=0 收缩为单点。"""
    return (
        neutral + (rng[0] - neutral) * scale,
        neutral + (rng[1] - neutral) * scale,
    )


def build_dr_events(scale: float) -> dict[str, EventTermCfg]:
    """按 scale 生成全部 DR 事件项；scale <= 0 时返回空字典（完全关闭）。"""
    if scale <= 0.0:
        return {}

    s = float(scale)
    # 四个脚共用同一套参数：分开采样会出现"三只脚不打滑、一只脚打滑"的
    # 非物理组合，而且随机化的目的是覆盖地面差异，不是制造脚间差异。
    foot_geoms = SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES)
    base_body = SceneEntityCfg("robot", body_names=("base_link",))
    leg_bodies = SceneEntityCfg(
        "robot", body_names=(".*_hip", ".*_thigh", ".*_calf")
    )
    all_joints = SceneEntityCfg("robot", joint_names=(".*",))

    return {
        # --- 接触参数：足底滑动摩擦 -------------------------------------
        # 真实地面的摩擦系数差别很大（瓷砖 ~0.4、橡胶垫 >1.0），这是
        # sim2real 最值得随机化的一项。
        "dr_foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": foot_geoms,
                "ranges": _shrink((0.3, 1.2), _FOOT_FRICTION_DEFAULT, s),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        # --- 机身质心：电池/线束造成的重心偏移 -------------------------
        # 只动 base_link 的 body_ipos，不改质量，也不与腿部伪惯量项冲突。
        "dr_base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": base_body,
                "operation": "add",
                "ranges": {
                    0: _shrink((-0.025, 0.025), _ZERO_NEUTRAL, s),
                    1: _shrink((-0.025, 0.025), _ZERO_NEUTRAL, s),
                    2: _shrink((-0.03, 0.03), _ZERO_NEUTRAL, s),
                },
            },
        ),
        # --- 腿部连杆质量/惯量 -----------------------------------------
        # 用伪惯量参数化同时缩放质量和转动惯量，保证物理一致；alpha 是
        # 对数尺度，质量与惯量按 e^(2*alpha) 缩放。只作用于腿：pseudo_inertia
        # 会重写 body_ipos，机身质心交给上面的 dr_base_com，两个事件不能
        # 同时作用在同一批 body 上。
        "dr_leg_inertia": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": leg_bodies,
                "alpha_range": _shrink((-0.1, 0.1), _ZERO_NEUTRAL, s),
            },
        ),
        # --- 关节级参数：电机/减速器的制造差异 -------------------------
        # 转子惯量（armature）和干摩擦（frictionloss）直接影响力矩响应，
        # 是 Go2 这类准直驱关节最主要的 sim2real gap。
        "dr_joint_armature": EventTermCfg(
            mode="startup",
            func=dr.joint_armature,
            params={
                "asset_cfg": all_joints,
                "ranges": _shrink((0.5, 1.5), _SCALE_NEUTRAL, s),
                "operation": "scale",
            },
        ),
        "dr_joint_damping": EventTermCfg(
            mode="startup",
            func=dr.joint_damping,
            params={
                "asset_cfg": all_joints,
                "ranges": _shrink((0.5, 1.5), _SCALE_NEUTRAL, s),
                "operation": "scale",
            },
        ),
        "dr_joint_friction": EventTermCfg(
            mode="startup",
            func=dr.joint_friction,
            params={
                "asset_cfg": all_joints,
                "ranges": _shrink((0.5, 2.0), _SCALE_NEUTRAL, s),
                "operation": "scale",
            },
        ),
        # --- 传感器：编码器零位偏差 -----------------------------------
        # 观测和执行器都读 joint_pos_biased，所以这个偏差同时影响策略
        # 输入和 PD 控制，而不是只改观测。
        "dr_encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": all_joints,
                "bias_range": _shrink((-0.015, 0.015), _ZERO_NEUTRAL, s),
            },
        ),
        # --- 外部扰动：随机推力 ---------------------------------------
        # 不属于参数随机化，但和 DR 一起用于提高鲁棒性。楼梯任务对横向
        # 推力很敏感，所以幅度和频率都比平地 velocity 任务更保守。
        "dr_push_robot": EventTermCfg(
            mode="interval",
            interval_range_s=(6.0, 10.0),
            func=envs_mdp.push_by_setting_velocity,
            params={
                "velocity_range": {
                    "x": _shrink((-0.4, 0.4), _ZERO_NEUTRAL, s),
                    "y": _shrink((-0.3, 0.3), _ZERO_NEUTRAL, s),
                    "z": _shrink((-0.2, 0.2), _ZERO_NEUTRAL, s),
                    "roll": _shrink((-0.3, 0.3), _ZERO_NEUTRAL, s),
                    "pitch": _shrink((-0.3, 0.3), _ZERO_NEUTRAL, s),
                    "yaw": _shrink((-0.5, 0.5), _ZERO_NEUTRAL, s),
                },
            },
        ),
    }
