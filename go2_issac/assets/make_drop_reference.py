"""MuJoCo 静置落地真值（本机可跑，只 import mujoco）。

    python3 -m go2_issac.assets.make_drop_reference
    python3 -m go2_issac.assets.make_drop_reference --print

把一个**零动作**的策略（`action ≡ 0`，也就是 `q_target = q_default`）从空中放下来，
记录 3 s 之后的稳态：机身高度、各关节稳态误差、脚底法向力合计、以及整段落地过程的高度曲线。

**为什么只有这个能当"物理参考"**：本机跑不了 Isaac，任何依赖接触动力学的量都没法在本地算出
Isaac 那边的值。但"给定同样的 PD、同样的质量、同样的初始高度，狗最终坐在离地多高"这件事，
两边应该**差不多**——差 1~2 cm 是接触模型差异（MuJoCo 的 `elliptic` 锥 + `condim=6` 脚底 vs
PhysX 的各向同性摩擦），差 5 cm 以上就是接线错了（kp/kd 写反、质量不对、地形没对上）。
所以上机后 `smoke.py --drop-test` 拿这份 JSON 比差值，而不是比相等。

四个工况刻意配成一对：
  * `flat_origin` / `flat_at_x340` —— 同一块平地、不同的 x。两者必须一致，
    否则说明地形高程图按 x 偏了（或高度场原点没对齐）。
  * `top_full_1.0` / `top_full_0.7` —— 站在 0.92 m 顶平台上，以及同一个 x 但地形缩到 0.7
    （顶面 0.644 m）。这两个的高度**差**就是高程图的 z 缩放，对得上的话
    `COURSE_ORIGIN_IN_PATCH` 和 `vertical_scale` 这两件事就一起确认了。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from go2_common.config import DEFAULT_Q, SIM_DT  # noqa: E402
from go2_common.env import Go2TerrainEnv  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "drop_reference.json"

# (名字, 地形, 缩放, x, y, 落下的初始高度)
CASES = (
    ("flat_origin", "flat", 1.0, 0.0, 0.0, 0.35),
    ("flat_at_x340", "flat", 1.0, 3.40, 0.0, 0.35),
    ("top_full_1.0", "full", 1.0, 3.40, 0.0, 0.35),
    ("top_full_0.7", "full", 0.7, 3.40, 0.0, 0.35),
)

HOLD_S = 3.0        # 静置时长
TAIL_S = 0.5        # 取稳态用的尾段（最后一秒的一半，足够滤掉落地余振）
TRACE_DT = 0.01     # 过程曲线的采样间隔（3 s -> 300 点）


def contact_breakdown(env: Go2TerrainEnv) -> tuple[float, float]:
    """(脚底法向力合计, 非脚部件撞地形的法向力合计)，单位 N。

    `env._contact_force()` 只算后者（奖励里惩罚的是"机身/腿蹭地形"）；脚底那部分这里单独要，
    因为静置时它必须等于整机重量——这是这类真值里唯一能自己验证自己的量。
    """
    import mujoco

    foot = 0.0
    body = 0.0
    res = np.zeros(6, dtype=np.float64)
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not (env.is_terrain_geom[g1] or env.is_terrain_geom[g2]):
            continue
        mujoco.mj_contactForce(env.model, env.data, i, res)
        f = abs(float(res[0]))
        if (env.is_terrain_geom[g1] and env.is_penalty_geom[g2]) or (
            env.is_terrain_geom[g2] and env.is_penalty_geom[g1]
        ):
            body += f
        else:
            foot += f
    return foot, body


def run_case(name: str, variant: str, scale: float, x: float, y: float,
             start_z: float) -> dict:
    import mujoco

    from go2_common.config import EnvCfg

    cfg = EnvCfg(terrain=variant, terrain_scale=scale)
    env = Go2TerrainEnv(cfg)

    # 确定性初始化：不用 env.reset()（那边是随机的），只借它的 home 关键帧与坐标约定
    d = env.data
    mujoco.mj_resetDataKeyframe(env.model, d, env._home_key_id)
    top_foot = float(env.terrain.top_footprint(x, y))
    d.qpos[0] = x
    d.qpos[1] = y
    d.qpos[2] = top_foot + start_z
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)

    steps = int(round(HOLD_S / SIM_DT))
    tail = int(round(TAIL_S / SIM_DT))
    trace_every = max(1, int(round(TRACE_DT / SIM_DT)))

    z_hist = np.zeros(steps)
    trace = []
    foot_last, body_last, z_last = 0.0, 0.0, 0.0
    q_err_last = np.zeros(env.model.nu)

    for k in range(steps):
        # 零动作 + PD，力矩算法与 env.step 完全一致（q_target = q_default）
        q_target = np.clip(env.q_default, env.q_lo, env.q_hi)
        for _ in range(cfg.decimation):
            q = d.qpos[env.qadr]
            dq = d.qvel[env.vadr]
            d.ctrl[:] = np.clip(cfg.kp * (q_target - q) - cfg.kd * dq,
                                env.tau_lo, env.tau_hi)
            mujoco.mj_step(env.model, d)

        z_hist[k] = d.qpos[2]
        if k >= steps - tail:
            foot, body = contact_breakdown(env)
            foot_last += foot / tail
            body_last += body / tail
            z_last += d.qpos[2] / tail
            q_err_last += (d.qpos[env.qadr] - env.q_default) / tail
        if k % trace_every == 0:
            trace.append([round(k * SIM_DT, 4), round(float(d.qpos[2]), 6),
                          round(float(d.qpos[2] - env.terrain.top(float(d.qpos[0]),
                                                          float(d.qpos[1]))), 6)])

    result = {
        "terrain": variant,
        "terrain_scale": scale,
        "start": {"x": x, "y": y, "z": start_z, "top_footprint": top_foot},
        "settle": {
            "base_pos": [round(float(v), 6) for v in d.qpos[:3]],
            "base_quat": [round(float(v), 6) for v in d.qpos[3:7]],
            "base_z": round(float(z_last), 6),
            # 唯一真正能跨仿真器比的量：机身**离当地地形**的高度（去掉了地形本身的差）
            "height_above_terrain": round(float(z_last - top_foot), 6),
            "joint_err": [round(float(v), 6) for v in q_err_last],   # 执行器顺序
            "joint_err_max": round(float(np.abs(q_err_last).max()), 6),
            "foot_force_N": round(float(foot_last), 4),
            "body_force_N": round(float(body_last), 4),
            "xy_drift": round(float(math.hypot(d.qpos[0] - x, d.qpos[1] - y)), 6),
        },
        "transient": {
            "z_max": round(float(z_hist.max()), 6),
            "z_min": round(float(z_hist.min()), 6),
            "trace_dt": TRACE_DT,
            # [t, base_z, 离地高度] 三元组；比稳态更多信息：能看出是"软着陆"还是"弹一下"
            "trace": trace,
        },
    }
    return result


def checks(ref: dict) -> list[str]:
    """自检：这些是物理上必须成立的，不成立说明建模/接线有问题，JSON 不能当真值。"""
    bad: list[str] = []
    weight = ref["robot"]["total_mass"] * abs(ref["robot"]["gravity_z"])
    flat_z = ref["cases"]["flat_origin"]["settle"]["base_z"]

    for name, case in ref["cases"].items():
        s = case["settle"]
        h = s["height_above_terrain"]

        # 1) 静置时脚底法向力合计 = 整机重量（牛顿第三定律，接触求解器必须满足）
        if abs(s["foot_force_N"] - weight) > 0.05 * weight:
            bad.append(f"{name}: 脚底力 {s['foot_force_N']:.2f} N 不等于重量 {weight:.2f} N")
        # 2) 站着的时候不该有"机身蹭地形"
        if s["body_force_N"] > 1.0:
            bad.append(f"{name}: 静置时非脚部件还有 {s['body_force_N']:.2f} N 接触力")
        # 3) 站姿高度应该落在合理区间（config 注释里的实测值是 0.268 m）
        if not 0.20 <= h <= 0.35:
            bad.append(f"{name}: 离地高度 {h:.4f} m 不在 [0.20, 0.35]")
        # 4) 没摔、没滑走
        if s["xy_drift"] > 0.05:
            bad.append(f"{name}: 静置漂了 {s['xy_drift']:.4f} m")
        # 5) 关节稳态误差小（PD 增益够硬）
        if s["joint_err_max"] > 0.25:
            bad.append(f"{name}: 关节稳态误差 {s['joint_err_max']:.4f} rad 过大")

    # 6) 同一块平地上，x=0 和 x=3.40 必须落到同一个高度
    dz = abs(ref["cases"]["flat_at_x340"]["settle"]["base_z"] - flat_z)
    if dz > 0.005:
        bad.append(f"平地 x=0 与 x=3.40 的稳态高度差 {dz:.4f} m（高程图按 x 偏了？）")

    # 7) 顶平台高度：1.0 档应比平地高约 0.92，0.7 档约 0.644
    for name, want in (("top_full_1.0", 0.92), ("top_full_0.7", 0.92 * 0.7)):
        got = ref["cases"][name]["settle"]["base_z"] - flat_z
        if abs(got - want) > 0.02:
            bad.append(f"{name}: 比平地高 {got:.4f} m，期望 {want:.4f} m")

    # 8) 最尖的一条：四个工况的"离地高度"必须一致。道理是——z 方向只平移不平移，
    #    地形只改脚下那 0.92/0.644 m，狗相对地表该怎么站还是怎么站。
    #    这条其实同时盖住了"高程图 z 缩放对不对"和"patch 原点有没有偏"两件事，
    #    容差给 0.1 mm（实测是逐位相同的 0.2676）。
    hs = {n: c["settle"]["height_above_terrain"] for n, c in ref["cases"].items()}
    spread = max(hs.values()) - min(hs.values())
    if spread > 1e-4:
        bad.append(f"四个工况的离地高度不一致，最大差 {spread:.6f} m：" +
                   "  ".join(f"{n}={h:.6f}" for n, h in hs.items()))

    return bad


def summarize(ref: dict) -> None:
    print(f"整机重量 {ref['robot']['total_mass']:.4f} kg × "
          f"{abs(ref['robot']['gravity_z']):g} = {ref['robot']['weight_N']:.2f} N")
    print(f"{'工况':<16} {'稳态 z':>9} {'离地高':>9} {'脚底力':>9} {'机身力':>8} "
          f"{'关节误差':>9} {'漂移':>7}")
    for name, c in ref["cases"].items():
        s = c["settle"]
        print(f"{name:<16} {s['base_z']:9.4f} {s['height_above_terrain']:9.4f} "
              f"{s['foot_force_N']:9.2f} {s['body_force_N']:8.3f} "
              f"{s['joint_err_max']:9.4f} {s['xy_drift']:7.4f}")
    print(f"\n{'关节稳态误差（执行器顺序 FR,FL,RR,RL × hip,thigh,calf）':<20}")
    for name, c in ref["cases"].items():
        e = np.asarray(c["settle"]["joint_err"])
        print(f"  {name:<16} " + " ".join(f"{v:+.4f}" for v in e))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(pathlib.Path(__file__).resolve().parents[2]
            / "unitree_mujoco" / "unitree_robots" / "go2" / "scene.xml"))
    gz = float(model.opt.gravity[2])

    ref = {
        "source": "Go2TerrainEnv，零动作（q_target=q_default）静置",
        "sim_dt": SIM_DT,
        "hold_s": HOLD_S,
        "q_default": [float(v) for v in DEFAULT_Q],
        "robot": {
            "total_mass": float(model.body_mass.sum()),
            "gravity_z": gz,
            "weight_N": float(model.body_mass.sum()) * abs(gz),
        },
        "cases": {},
    }
    for name, variant, scale, x, y, z0 in CASES:
        ref["cases"][name] = run_case(name, variant, scale, x, y, z0)
        print(f"  {name} 完成")

    # 上机后 `smoke.py --drop-test` 真正要比的就是这一个数：
    # "零动作站定时，机身离当地地形多高"。MuJoCo 这边 0.2676 m。
    ref["summary"] = {
        "settle_height_above_terrain": ref["cases"]["flat_origin"]["settle"]["height_above_terrain"],
        "how_to_compare": "Isaac 侧同样零动作静置 3 s，比 height_above_terrain；"
                          "差 1~2 cm 是接触模型差异，> 5 cm 是接线错了。",
    }

    bad = checks(ref)
    if bad:
        print("\n自检失败（物理上说不通，别把 JSON 当真值用）：")
        for b in bad:
            print(f"  - {b}")
        return 1

    out = pathlib.Path(args.out)
    with open(out, "w") as f:
        json.dump(ref, f, indent=1)
        f.write("\n")
    print(f"\n已写出 {out}（{out.stat().st_size / 1024:.0f} KB），自检通过")
    if args.do_print:
        print()
        summarize(ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
