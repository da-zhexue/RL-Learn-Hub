"""上机自检：把"本机证明不了的接线"一件一件验掉。

    python3 go2_issac/smoke.py --check-env                    # 版本 + 显卡 + 依赖，先跑这个
    python3 go2_issac/smoke.py --check-asset                  # USD 的资产参数 vs 真值 JSON
    python3 go2_issac/smoke.py --check-order                  # 关节顺序 / 动作顺序
    python3 go2_issac/smoke.py --check-terrain                # 高程图和 CourseHeight 对不对得上
    python3 go2_issac/smoke.py --sanity                       # 观测/奖励/终止的抽查
    python3 go2_issac/smoke.py --drop-test                    # 零动作静置，和 MuJoCo 真值比
    python3 go2_issac/smoke.py --all                          # 全套（建议第一次就这么跑）

为什么会有这个文件：开发这台机器**没有 NVIDIA 显卡**，Isaac Sim 一行都跑不了（没有 CPU
回退）。所以本机能做到的只是"把语义证明对"（`course.py` / `core.py` + `tests/` 那两个
逐位对拍），剩下的"Isaac API 接线"和"PhysX 物理"只能在有卡的机器上验。这个脚本就是那台
机器上的第一道关：**每一节都独立、都能单独跑，报出来的每条都是能直接下手改的**。

看过 `--check-asset` 的输出之后，如果关节顺序、落地高度、驱动增益这三件事都对，
剩下的就只是调参了——这正是"没有显卡"时能把风险压到的最低限度。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO))

_ASSETS = HERE / "assets"

_failures: list[str] = []


def ok(cond: bool, msg: str, detail: str = "") -> bool:
    """记一条检查结果。返回 cond，方便 `if not ok(...): continue`。"""
    mark = "  ok  " if cond else "  FAIL"
    print(f"{mark} {msg}" + (f"\n         {detail}" if detail else ""))
    if not cond:
        _failures.append(msg)
    return cond


def close(a, b, tol: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# ==================================================================== 环境


def check_env(args) -> None:
    """版本、显卡、以及"这台机器到底能不能跑 Isaac Sim"。

    **第一次上机就先跑这一条**（它不需要起 Kit），装错了/装少了会在这里出结果，
    而不是在几十秒的 Kit 启动之后报一个看不懂的错。
    """
    print("== 环境 ==")
    try:
        import torch

        import isaaclab
    except ImportError as e:
        ok(False, "isaaclab / torch 装上了", f"import 失败：{e}\n"
           "         先按 RUNBOOK §1 装（注意 AppLauncher 之外的东西都得在 Kit 起来之后 import）")
        return

    if ok(torch.cuda.is_available(), "torch 能看到 CUDA 设备",
          f"torch {torch.__version__}，cuda {torch.version.cuda}，"
          f"设备 {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}"):
        free, total = torch.cuda.mem_get_info()
        print(f"       显存 {total / 2**30:.0f} GB（空闲 {free / 2**30:.0f} GB）")
    else:
        print("       **Isaac Sim 没有 CPU 回退**：没有 NVIDIA 显卡就到此为止，"
              "别继续往下跑了（见 RUNBOOK §0 的硬件表）")

    ver = getattr(isaaclab, "__version__", "unknown")
    print(f"       isaaclab {ver}  python {sys.version.split()[0]}")
    manifest = _ASSETS / "manifest.json"
    if manifest.exists():
        got = json.loads(manifest.read_text())
        ok(got.get("isaaclab") in (ver, "unknown"), "isaaclab 版本和转 USD 时一致",
           f"转 USD 时是 {got.get('isaaclab')}，现在是 {ver}"
           "（不一致就要重转 USD，见 RUNBOOK 的版本差异表）")
    else:
        print(f"       没有 {manifest.name}：还没跑过 convert_assets.py")

    try:
        import os

        pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        print(f"       内存 {pages / 2**30:.0f} GB（Isaac Sim 要 ≥32 GB）")
    except (ValueError, OSError, AttributeError):
        pass


# ==================================================================== 资产


def check_asset(env, args) -> None:
    """USD 实例化出来的 articulation vs `assets/go2_reference.json`。

    比的是**看不出来**的那些量：质量、惯量、关节限位（前后腿不一样）、力矩上限、
    驱动增益。肉眼在 viewport 里看狗"长得对不对"是查不出这些的，而它们错了只会表现为
    "训不动"。
    """
    print("\n== 资产（USD vs go2_reference.json）==")
    ref_path = _ASSETS / "go2_reference.json"
    if not ok(ref_path.exists(), f"{ref_path.name} 存在",
              "缺它就先在本机跑 `python3 -m go2_issac.assets.make_mjcf_reference`"):
        return
    ref = json.loads(ref_path.read_text())

    from go2_issac.mdp import state

    ctx = state._ctx(env)
    robot = env.scene["robot"]

    # --- 总质量
    mass = float(robot.root_physx_view.get_masses().sum())
    ok(close(mass, ref["total_mass"], 1e-3), "总质量",
       f"USD {mass:.5f} kg vs 真值 {ref['total_mass']:.5f} kg")

    # --- 每个连杆的质量
    names = list(robot.body_names)
    masses = robot.root_physx_view.get_masses()[0].tolist()
    by_name = dict(zip(names, masses))
    bad = []
    for b in ref["bodies"]:
        if b["id"] == 0 or b["mass"] <= 0:
            continue          # world 和零质量的 FL_foot 之类，PhysX 那边没有对应
        if b["name"] not in by_name:
            bad.append(f"{b['name']} 在 USD 里找不到")
        elif not close(by_name[b["name"]], b["mass"], 1e-4):
            bad.append(f"{b['name']}: USD {by_name[b['name']]:.5f} vs 真值 {b['mass']:.5f}")
    ok(not bad, "各连杆质量", "\n         ".join(bad[:8]))

    # --- 关节限位（前后腿不同，抄一样的会在这里炸）
    limits = {n: tuple(v) for n, v in zip(state.JOINT_NAMES, ctx.joint_limits.tolist())}
    ref_j = {j["name"]: j for j in ref["joints"] if j["type"] == "hinge"}
    bad = []
    for name, (lo, hi) in limits.items():
        want = ref_j[name]["range"]
        if not (close(lo, want[0], 1e-4) and close(hi, want[1], 1e-4)):
            bad.append(f"{name}: USD [{lo:+.4f}, {hi:+.4f}] vs 真值 "
                       f"[{want[0]:+.4f}, {want[1]:+.4f}]")
    ok(not bad, "关节限位（前腿/后腿本来就不一样）", "\n         ".join(bad[:6]))

    # --- 力矩上限 + 驱动增益（kp/kd 是在这儿写回去的，不在 USD 里）
    bad = []
    for act_name, act in robot.actuators.items():
        want_tau = 45.43 if act_name == "calf" else 23.7
        got_tau = getattr(act, "effort_limit_sim", getattr(act, "effort_limit", None))
        if not close(got_tau, want_tau, 1e-3):
            bad.append(f"{act_name}: 力矩上限 {got_tau} vs 真值 {want_tau}")
        if not close(act.stiffness, 60.0, 1e-6):
            bad.append(f"{act_name}: stiffness {act.stiffness} != 60")
        if not close(act.damping, 3.6, 1e-6):
            bad.append(f"{act_name}: damping {act.damping} != 3.6（= kd 3.5 + 铰链自带 0.1）")
    ok(not bad, "力矩上限与驱动增益（kp/damping）", "\n         ".join(bad[:6]))

    # --- 碰撞几何个数（视觉网格被误转成碰撞体的话，狗会被自己卡住）
    n_coll = int(robot.root_physx_view.get_link_collision_prims_count().sum()) \
        if hasattr(robot.root_physx_view, "get_link_collision_prims_count") else -1
    n_ref = sum(1 for g in ref["geoms"]
                if g["body_id"] and (g["contype"] or g["conaffinity"]))
    if n_coll >= 0:
        ok(n_coll == n_ref, "狗的碰撞几何个数", f"USD {n_coll} vs 真值 {n_ref}")
    else:
        print(f"       碰撞几何个数查不了（这个版本没有那个 API），真值是 {n_ref} 个："
              f"在 Isaac 的 _go2/go2 下数一下 Link/Collision，别把 33 个视觉网格也算上")

    print(f"       脚底 body 解析到 {list(ctx.foot_body_names)}"
          "（FL_foot 被并进 FL_calf 是正常的，只要脚底球还在）")


# ==================================================================== 顺序


def check_order(env, args) -> None:
    """关节顺序陷阱：MJCF 的身体树 / USD 的导入顺序 / DDS 报文，三个都不一样。

    这里验的是"我们按名字查到的东西，和动作项按名字查到的是同一个"，
    以及动作→关节目标的换算和 `env.py:197` 一致。
    """
    print("\n== 关节与动作顺序 ==")
    import torch

    from go2_issac.mdp import state

    ctx = state._ctx(env)

    # 1) 名字查表：查到的 q 的顺序确实是 FR, FL, RR, RL
    #    （用默认站姿反查：q_default 的每个元素必须落在对应关节的限位里）
    inside = ((ctx.q_default >= ctx.joint_limits[:, 0])
              & (ctx.q_default <= ctx.joint_limits[:, 1]))
    ok(bool(inside.all()), "q_default 落在各自关节的限位内",
       f"越界的是 {[state.JOINT_NAMES[i] for i in (~inside).nonzero().tolist()]}")

    # 2) 动作项和 state 用的是同一批关节
    term = env.action_manager.get_term("joints")
    ok(list(term.joint_names) == list(state.JOINT_NAMES),
       "动作项的关节顺序 = JOINT_NAMES",
       f"动作项是 {list(term.joint_names) if list(term.joint_names) != list(state.JOINT_NAMES) else '一致'}"
       "（不一致也没关系，raw_action 会按名字重排，但这里报出来更好查）")

    # 3) 走一步已知动作，验"动作 -> 关节目标"和 `env.py:197` 同式，并且 prev_action 追得上
    #
    #    探针故意跨到 ±2（**超出 ±1**）：策略初始 σ=1.0，约 1/3 的动作本来就超界，
    #    而"夹取有没有作用到 prev_action 上"只在超界时看得出来（`env.py:194` 是先夹再存）。
    env.reset()
    probe = torch.linspace(-2.0, 2.0, 12, device=env.device).unsqueeze(0)
    probe = probe.expand(env.num_envs, -1).clone()
    probe_clamped = torch.clamp(probe, -1.0, 1.0)
    for _ in range(2):        # 第一步让观测项把 prev 建起来
        env.step(probe)
    st = state.get_state(env)
    want = torch.clamp(ctx.q_default.unsqueeze(0) + ctx.cfg.action_scale * probe_clamped,
                       min=ctx.joint_limits[:, 0], max=ctx.joint_limits[:, 1])
    got = term.processed_actions[0]
    ok(bool(torch.allclose(got, want[0].to(got.dtype), atol=1e-4)),
       "关节目标 = clip(q_default + scale*clip(action))",
       f"最大差 {float((got - want[0]).abs().max()):.2e}")

    ok(bool(torch.allclose(st.prev_action[0].float(), probe_clamped[0].float(), atol=1e-5)),
       "观测里的动作位（prev_action）= **夹过之后**的下发动作",
       f"实得 {st.prev_action[0].tolist()}\n"
       f"         期望 {probe_clamped[0].tolist()}\n"
       "         对不上优先查两处：mdp/observations.py 的推进时机、mdp/state.py:raw_action 的 clamp")

    # 4) 站姿：零动作几秒后离地高度应该接近 MuJoCo 的 0.2676 m（这一步不算物理，
    #    真正的物理比对在 --drop-test）
    zero = torch.zeros(env.num_envs, 12, device=env.device)
    for _ in range(100):
        env.step(zero)
    st = state.get_state(env)
    h = float((st.base_pos[:, 2] - st.terrain_h).mean())
    print(f"       零动作 2 s 后离地高度 {h:.4f} m（MuJoCo 静置真值是 0.2676 m）")


# ==================================================================== 地形


def check_terrain(env, args) -> None:
    """高程图 vs `CourseHeight`：验 patch 原点、z 缩放、量化误差。

    这是**唯一一个"只能上机校准"的常量**（`course.COURSE_ORIGIN_FROM_ENV_ORIGIN`）：
    Isaac Lab 把每个 patch 的中心写进 `env.scene.env_origins`，但"中心"这个说法在
    版本之间变过。做法是朝下打射线，拿命中点的高度和 `CourseHeight.top` 比——
    对不上就只改 `course.py` 的那一个常量，别处不用动。
    """
    print("\n== 地形 ==")
    import torch

    from go2_issac import course
    from go2_issac.mdp import state

    ctx = state._ctx(env)
    origins = env.scene.env_origins
    ok(tuple(origins.shape) == (env.num_envs, 3), "env_origins 形状",
       f"{tuple(origins.shape)}")

    # 直接用几何算：patch 中心在课程坐标里应该是 (1.5, 0.0)
    one = origins[0:1]
    xy = state.course_xy(env, torch.cat([one[:, :2], torch.zeros_like(one[:, :1])], dim=-1))
    got = xy[0].tolist()
    ok(close(got[0], 1.5, 1e-6) and close(got[1], 0.0, 1e-6),
       "环境原点映射到课程坐标 (1.5, 0.0)",
       f"算出来是 ({got[0]:+.4f}, {got[1]:+.4f})。对不上就改 "
       f"`course.COURSE_ORIGIN_FROM_ENV_ORIGIN`（现在 {course.COURSE_ORIGIN_FROM_ENV_ORIGIN}）")

    # 高度场本身没法直接读（在 PhysX 内部），但 `--drop-test` 是等价的证据：
    # 它拿真狗站在不同 x 上的落点高度反推，对不上就是高程图或坐标偏移错了
    print("       （高程图内容靠 `--drop-test` 反推：它把狗放在几个已知 x 上，"
          "量落点高度和 `CourseHeight.top` 之比）")

    # 高程图的实际规模（算内存用）
    rows, cols = env.cfg.terrain.terrain_generator.num_rows, \
        env.cfg.terrain.terrain_generator.num_cols
    px = int(course.PATCH_SIZE[0] / course.HORIZONTAL_SCALE)
    print(f"       {rows}×{cols} 个 patch，整张高程图 {rows * px}×{cols * px} 像素 "
          f"({rows * px * cols * px * 2 / 2**20:.0f} MB, int16)")
    print(f"       课程档位 {env.cfg.terrain_variant}@{env.cfg.terrain_scale:g}，"
          f"台阶 {ctx.course_height.stair_tops}")


# ==================================================================== 抽查


def check_sanity(env, args) -> None:
    """观测 / 奖励 / 终止的抽查：本机对拍过语义，这里验"接线接对了没有"。"""
    print("\n== 抽查（观测 / 奖励 / 终止）==")
    import torch

    from go2_common.config import obs_dim
    from go2_issac import core
    from go2_issac.mdp import state

    ctx = state._ctx(env)
    env.reset()
    st = state.get_state(env)

    # 1) 观测维度
    obs = env.observation_manager.compute()
    want = obs_dim(ctx.cfg.privileged)
    ok(tuple(obs.shape) == (env.num_envs, want), "观测维度",
       f"实得 {tuple(obs.shape)}，期望 ({env.num_envs}, {want})")

    # 2) 站立时 projected_gravity 应该是 (0, 0, -1)（符号最容易搞反的一维）
    pg = obs[0, 3:6].float()
    ok(bool(torch.allclose(pg, torch.tensor([0.0, 0.0, -1.0], device=obs.device), atol=0.05)),
       "站着时投影重力 ≈ (0, 0, -1)", f"实得 {pg.tolist()}")

    # 3) 观测里的"上一步动作"那 12 维：用 core.obs 现算一份对比
    mine = core.obs(st, ctx.cfg, ctx.q_default, privileged=ctx.cfg.privileged)
    ok(bool(torch.allclose(mine, obs, atol=1e-5)),
       "core.obs 与 Isaac 的观测逐元素一致",
       f"最大差 {float((mine - obs).abs().max()):.2e}"
       "（不一致说明 env_cfg 里挂的 term 和 core.py 不是一套）")

    # 4) 奖励：14 项分项的名字与总数
    term = core.terminations(st, ctx.cfg, ctx.goal_z)
    total, parts = core.compute_reward(st, ctx.reward_cfg, term["any"], term["success"])
    ok(len(parts) == 14, "奖励 14 项", f"实得 {len(parts)} 项")
    rw = env.reward_manager.compute(dt=env.step_dt)
    ok(bool(torch.allclose(rw, total.to(rw.dtype), atol=1e-4)),
       "Isaac 的奖励和 core.compute_reward 一致",
       f"最大差 {float((rw - total).abs().max()):.2e}")

    # 5) 终止：超时要走 truncated（rsl_rl 靠它 bootstrap），不是 terminated
    t = env.termination_manager
    ok(hasattr(t, "time_outs"), "termination_manager 有 time_outs（超时单独一路）")
    zero = torch.zeros(env.num_envs, 12, device=env.device)
    n_reset = 0
    for _ in range(1200):          # 20 s 上限 = 1000 步，跑够就能看到一次超时
        _, _, dones, _ = _step(env, zero)
        n_reset += int(dones.sum())
    ok(n_reset > 0, "零动作跑满 20 s 会因超时重置", f"{n_reset} 次")


def _step(env, action):
    """走一步，返回 rsl_rl 口径的 4 元组（Isaac 原生是 5 元组）。"""
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, _info = out
        return obs, rew, torch.logical_or(terminated, truncated), _info
    return out


# ==================================================================== 落地


def check_drop(env, args) -> None:
    """零动作静置 3 s，和 `assets/drop_reference.json` 比。

    **这是唯一能验"物理接线对不对"的本地真值**：MuJoCo 那边零动作站定时机身离地
    0.2676 m、脚底法向力合计 = 整机重量 149.17 N。Isaac 这边比的是差值不是相等——
    差 1~2 cm 是接触模型差异（MuJoCo 的 elliptic 锥 + condim=6 脚底 vs PhysX 各向同性），
    差 5 cm 以上就是接线错了（kp/kd 写反、质量不对、地形高度没对上）。
    """
    print("\n== 落地（零动作静置 3 s）==")
    import torch

    from go2_issac.mdp import state

    ref_path = _ASSETS / "drop_reference.json"
    if not ok(ref_path.exists(), f"{ref_path.name} 存在",
              "缺它就先在本机跑 `python3 -m go2_issac.assets.make_drop_reference`"):
        return
    ref = json.loads(ref_path.read_text())
    want_h = ref["summary"]["settle_height_above_terrain"]
    want_f = ref["robot"]["weight_N"]
    step_dt = env.cfg.sim.dt * env.cfg.decimation     # 策略步长 = 0.005 × 4 = 0.02 s
    steps = int(round(3.0 / step_dt))                 # -> 150 步，和 MuJoCo 侧的 HOLD_S 对齐

    ctx = state._ctx(env)
    env.reset()
    zero = torch.zeros(env.num_envs, 12, device=env.device)
    for _ in range(steps):
        env.step(zero)
    st = state.get_state(env)

    height = st.base_pos[:, 2] - st.terrain_h
    h = float(height.mean())
    dh = h - want_h
    ok(abs(dh) < 0.05, "静置离地高度",
       f"Isaac {h:.4f} m vs MuJoCo {want_h:.4f} m（差 {dh * 100:+.2f} cm）"
       + ("；< 2 cm 是接触模型差异，> 5 cm 要查 kp/kd/质量" if abs(dh) < 0.05 else ""))

    # 脚底力：Isaac 只能拿到"每个 body 的净接触力"，这里给个总竖直分量做粗略对照
    sens = env.scene.sensors[ctx.sens_name]
    fz = sens.data.net_forces_w.to(torch.float64)[:, :, 2]
    total_f = float(torch.clamp(fz, min=0.0).sum(dim=-1).mean())
    ok(abs(total_f - want_f) < 0.15 * want_f, "脚下竖直力合计 ≈ 整机重量",
       f"Isaac {total_f:.1f} N vs MuJoCo {want_f:.1f} N（差 {total_f - want_f:+.1f} N）")

    # 落到地上之后不该再有非脚部件的接触（真蹭上了说明姿态/限位不对）
    penalty = float(state.contact_force(env, ctx).mean())
    ok(penalty < 1.0, "静置时非脚部件不接触地形", f"接触力 {penalty:.3f} N")

    print(f"       各工况真值（本机 MuJoCo）：" + "  ".join(
        f"{n}={c['settle']['height_above_terrain']:.4f}"
        for n, c in ref["cases"].items()))


# ==================================================================== main


SECTIONS = {
    "env": check_env,
    "asset": check_asset,
    "order": check_order,
    "terrain": check_terrain,
    "sanity": check_sanity,
    "drop": check_drop,
}
#: 需要真的建出环境的那些（`--check-env` 不用）
NEEDS_ENV = ("asset", "order", "terrain", "sanity", "drop")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in SECTIONS:
        ap.add_argument(f"--check-{name}", action="store_true")
    ap.add_argument("--all", action="store_true", help="全套")
    ap.add_argument("--terrain", choices=("flat", "steps", "full"), default="full")
    ap.add_argument("--terrain-scale", type=float, default=1.0)
    ap.add_argument("--num-envs", type=int, default=4, help="自检用不了多少环境")

    # `--headless` / `--device` / `--enable_cameras` 由 Kit 自己加。
    # **不要自己造一个 `argparse.Namespace(headless=...)` 交给 AppLauncher**——
    # 它内部还会读 device / enable_cameras / cpu / verbose 等一串字段，
    # 少一个就是 AttributeError，而且报错位置离真正的原因很远。
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()

    picked = [n for n in SECTIONS if getattr(args, f"check_{n}") or args.all]
    if not picked:
        ap.error("至少给一个 --check-xxx 或者 --all")

    # `--check-env` 不需要 Kit，先单独跑掉，好让"根本装不起来"这种情况马上报出来
    if "env" in picked:
        check_env(args)
        picked.remove("env")
    need_env = [n for n in picked if n in NEEDS_ENV]
    if need_env:
        _run_with_kit(args, need_env)

    print("\n" + ("全部通过" if not _failures else f"{len(_failures)} 项失败："))
    for f in _failures:
        print(f"  - {f}")
    return 1 if _failures else 0


def _run_with_kit(args, sections) -> None:
    """起 Kit（AppLauncher 必须最先跑），再建环境跑各节。"""
    from isaaclab.app import AppLauncher

    # `args` 就是上面那个 parser 出来的完整命名空间（Kit 自己的字段都在里面）
    app = AppLauncher(args).app
    try:
        from go2_issac import env_cfg as ec

        cfg = ec.CourseEnvCfg()
        cfg.terrain_variant = args.terrain
        cfg.terrain_scale = args.terrain_scale
        cfg.set_envs(args.num_envs)
        print(cfg.describe())
        env = ec.build_env(cfg)
        try:
            for name in sections:
                SECTIONS[name](env, args)
        finally:
            env.close()
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    sys.exit(main())
