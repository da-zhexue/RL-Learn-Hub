"""诊断工具：地形、步态、策略卡住时到底发生了什么。

    python3 -m go2_sac.diag terrain     # 地形几何 & 课程缩放是否正确
    python3 -m go2_sac.diag stance      # 标称站姿能站多高、肚子离地多少（决定爬不爬得过槛）
    python3 -m go2_sac.diag lift        # 手调小跑的抬脚高度 -> 能跨多高的槛
    python3 -m go2_sac.diag frontier    # 扫 action_scale × 槛高，量出可跨边界
    python3 -m go2_sac.diag policy <model.zip> [地形] [scale]   # 训练后策略的抬脚高度
    python3 -m go2_sac.diag why <model.zip> <scale>            # 回合为什么提前结束
    python3 -m go2_sac.diag reward <model.zip> <scale>         # 分项奖励拆解

`stance` 回答的是另一个问题：不是"脚能抬多高"，而是"**肚子能离地多高**"。
狗卡在槛前不是抬不起脚（前脚实测能抬 0.154 m），是肚子比槛还低——
平地走路时 base_link 离地只有 0.065 m，而槛高 0.08 m，前脚搭上去之后肚子顶在槛面上。
而肚子能有多高，上限由标称站姿 DEFAULT_Q 和 action_scale 一起锁死。
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

from go2_common.config import DEFAULT_SCENE, EnvCfg, RewardCfg, quat_to_rpy
from go2_common.env import FOOT_GEOM_NAMES, Go2TerrainEnv, run_episode
from go2_common.train_utils import load_resume_configs
from go2_sac.train import trot_action


# ------------------------------------------------------------------ 通用


def load_policy(path):
    from stable_baselines3 import SAC

    return SAC.load(path, device="cpu")


def cfg_from_model(path):
    """按模型自带的 config.json 还原 EnvCfg。

    这几个子命令原来都把 action_scale 写死成 0.6，出生点也没设。拿它去量
    A=0.7 训出来的模型，环境会用 0.6 去缩放它的动作——量到的根本不是那个策略，
    会得出"脚抬不到 0.095"这种错误结论。reset_x 同理：训练时出生点在楼梯口，
    评估时却从起点出发，一个 20 秒回合里大半时间在平地上走。
    """
    cfg, _ = load_resume_configs(pathlib.Path(path))
    return cfg if cfg is not None else EnvCfg()


def foot_ids(env):
    return {n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES}


def run_once(env, act, seed=0):
    """跑一个回合，返回 (最远x, 最高台阶, 步数, info)。"""
    st = {"x": -9.0, "lv": 0}

    def on_step(e):
        x, y = float(e.data.qpos[0]), float(e.data.qpos[1])
        st["x"] = max(st["x"], x)
        st["lv"] = max(st["lv"], e.terrain.level(x, y))

    total, info = run_episode(env, act, seed=seed, on_step=on_step)
    return st["x"], st["lv"], total, info


def policy_act(model):
    return lambda o, e: model.predict(o, deterministic=True)[0]


# ------------------------------------------------------------------ 地形


def cmd_terrain(argv):
    from go2_common import terrain as T

    print("地形几何（每个方块的顶面高度）与课程缩放：\n")
    for name, scale in (("flat", 1.0), ("steps", 1.0), ("steps", 0.35),
                        ("full", 1.0), ("full", 0.55), ("full", 0.35)):
        m = T.build_model(DEFAULT_SCENE, name, scale)
        t = T.TerrainHeight(m)
        tops = sorted({round(b[4], 4) for b in t.boxes})
        print(f"  {name:>5} scale={scale:<5} 方块{len(t.boxes):>2} 个  台阶{len(t.stair_tops)} 级  "
              f"顶面={tops}")
    print("\n成功高度 goal_z（由 env 按地形算出，不是写死的）：")
    for name, scale in (("flat", 1.0), ("steps", 1.0), ("full", 0.35), ("full", 1.0)):
        env = Go2TerrainEnv(EnvCfg(terrain=name, terrain_scale=scale))
        print(f"  {name:>5} scale={scale:<5} 平台顶={env.terrain.top(3.3):.4f}  "
              f"goal_z={env.cfg.goal_z}")
        env.close()
    print("\n没有台阶的课程 goal_z 必须是 inf —— 否则 flat 上'走到 x>=3.3'就判成功，")
    print("回合提前结束，实测平地回报会从 715 掉到 281。")
    return 0


# ------------------------------------------------------------------ 标称站姿


def lowest_points(env):
    """每个几何体最低点的世界 z 坐标。

    不能拿 geom_xpos[2] 当"肚子高度"——那是几何体**中心**。躯干是 mesh，
    中心离底面还有半个身位。这里按类型算真实最低点：mesh 取顶点、box 取支撑函数，
    剩下的用外接球半径兜底（保守，只会低估不会高估）。
    """
    m, d = env.model, env.data
    out = np.empty(m.ngeom)
    for g in range(m.ngeom):
        c, sz, xmat = d.geom_xpos[g], m.geom_size[g], d.geom_xmat[g].reshape(3, 3)
        t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            a = m.mesh_vertadr[m.geom_dataid[g]]
            n = m.mesh_vertnum[m.geom_dataid[g]]
            out[g] = c[2] + (m.mesh_vert[a:a + n] @ xmat.T)[:, 2].min()
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            out[g] = c[2] - float(np.abs(xmat[2]) @ sz)
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out[g] = c[2] - sz[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            out[g] = c[2] - (sz[0] + sz[1])
        else:
            out[g] = c[2] - m.geom_rbound[g]
    return out


def belly_clearance(env) -> float:
    """躯干（base_link）最低点离当地地形的高度。

    **只能量躯干**。量"除脚以外的所有几何体"是错的：那个最小值永远被小腿占着
    （小腿本来就贴地），量出来会一直是个接近 0 的数，看着像"肚子比槛还低"，
    实际上躯干在 base z=0.27 时最低点有 0.163 —— 比 0.08 的槛高得多。
    真正卡住狗的是小腿/脚，不是肚子。躯干最低点是前端那个 r=0.047、z=-0.06 的球。
    """
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    lo = lowest_points(env)
    gs = [g for g in range(env.model.ngeom) if env.model.geom_bodyid[g] == bid]
    h = env.terrain.top(float(env.data.qpos[0]), float(env.data.qpos[1]))
    return float(min(lo[g] for g in gs)) - h


def hold_stance(env, q_nom, action, settle=60, measure=25):
    """把狗按 q_nom 摆好，施加常值 action 站一会儿，返回稳态的 (base_z, 肚子离地)。

    站立判据：机身高度几乎不动、关节速度很小。不满足就返回 None，表示这个姿势站不住。
    """
    d = env.data
    d.qpos[:] = 0.0
    d.qpos[0:3] = [0.0, 0.0, 0.34]
    d.qpos[3] = 1.0
    d.qpos[env.qadr] = q_nom
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)

    for _ in range(settle):
        env.step(action)
    zs, clr, dq = [], [], []
    for _ in range(measure):
        env.step(action)
        zs.append(float(d.qpos[2]))
        clr.append(belly_clearance(env))
        dq.append(float(np.abs(d.qvel[env.vadr]).max()))
    if np.std(zs) > 0.005 or max(dq) > 1.0 or np.mean(zs) < 0.15:
        return None
    return float(np.mean(zs)), float(np.min(clr))


def stance_scan(nominal, action_scale, n_actions=200, seed=0):
    """随机撒 200 个常值动作，量出这套标称站姿/幅度能稳定站住的 base z 与肚子离地范围。

    nominal 只写一条腿的 [hip, thigh, calf]，四条腿都一样（Go2 默认站姿就是对称的）。
    """
    from go2_common import env as E

    one = np.asarray(nominal, dtype=np.float64).ravel()
    if one.size == 3:
        one = np.tile(one, 4)
    old = E.DEFAULT_Q
    E.DEFAULT_Q = one
    try:
        env = Go2TerrainEnv(EnvCfg(terrain="flat", action_scale=action_scale), RewardCfg())
        rng = np.random.default_rng(seed)
        q_nom = env.q_default
        base, belly = [], []
        for _ in range(n_actions):
            got = hold_stance(env, q_nom, rng.uniform(-1.0, 1.0, env.model.nu))
            if got:
                base.append(got[0])
                belly.append(got[1])
        env.close()
    finally:
        E.DEFAULT_Q = old
    return np.array(base), np.array(belly)


def cmd_stance(argv):
    from go2_common.config import DEFAULT_Q

    print(f"当前 DEFAULT_Q = {np.array(DEFAULT_Q).reshape(4, 3).tolist()}  "
          f"action_scale = {EnvCfg().action_scale}\n")
    print("标称站姿 / 幅度  ->  能稳定站住的 base z 范围（肚子离地）：")
    rows = []
    if not argv:
        cands = [
            (list(DEFAULT_Q[:3]), EnvCfg().action_scale),
            ([0.0, 0.75, -1.35], 0.5),
            ([0.0, 0.7, -1.2], 0.5),
            ([0.0, 0.65, -1.05], 0.5),
            ([0.0, 0.7, -1.2], 0.6),
            ([0.0, 0.7, -1.2], 0.7),
        ]
    else:
        cands = [([0.0, float(argv[0]), float(argv[1])], float(argv[2]))]
    for nom, sc in cands:
        base, belly = stance_scan(nom, sc)
        tag = " <-当前" if list(nom) == list(np.array(DEFAULT_Q[:3])) and sc == EnvCfg().action_scale else ""
        if len(base):
            rows.append((nom, sc, len(base), base.min(), base.max(), belly.min(), belly.max()))
            print(f"  {str(nom):<22} {sc:<4} n={len(base):<4} "
                  f"base z {base.min():.3f}~{base.max():.3f}   "
                  f"肚子 {belly.min():+.3f}~{belly.max():+.3f}{tag}")
        else:
            print(f"  {str(nom):<22} {sc:<4} 站不住{tag}")

    print("\n槛高 0.08 m：**肚子最低点**要能超过 0.08 才过得去（不是脚，脚抬得起没用）。")
    print("走起来的实际姿态由 RewardCfg.height_target 决定，它把狗按在哪个高度，")
    print("狗就一直在那个高度走 —— 所以 height_target 必须落在上表肚子里程之内，")
    print("且留出余量：贴着可达上限走，动作一半的行程都浪费在'够不到'上。")
    return 0


# ------------------------------------------------------------------ 抬脚高度


def cmd_lift(argv):
    """手调开环小跑的抬脚高度随幅度的变化。

    `skip` 之后才统计：reset 的落地过程里四条腿全是悬空的，量出来的不是步态（见 AGENT.md §5-C5）。
    """
    skip = int(argv[0]) if argv else 150
    ascale = float(argv[1]) if len(argv) > 1 else EnvCfg().action_scale
    print(f"开环小跑的脚底球心离地间隙（球半径 0.022 m），只统计第 {skip} 步之后的稳定行走段：\n")
    print(f"action_scale = {ascale:g}\n")
    print(f"{'动作幅度':>8} {'绝对摆幅':>9} | " + " ".join(f"{n:>18}" for n in FOOT_GEOM_NAMES))
    for amp_u in (0.0, 0.3, 0.5, 0.7, 1.0):
        env = Go2TerrainEnv(EnvCfg(terrain="flat", action_scale=ascale), RewardCfg())
        ids, tr, hold = foot_ids(env), {n: [] for n in FOOT_GEOM_NAMES}, {"t": 0}

        def on_step(e, _ids=ids, _tr=tr, _skip=skip):
            if e.step_count < _skip:
                return
            g = e.terrain.top(float(e.data.qpos[0]), float(e.data.qpos[1]))
            for n, i in _ids.items():
                _tr[n].append(float(e.data.geom_xpos[i][2]) - g)

        def act(o, e, _amp=amp_u, _hold=hold):
            a = trot_action(_hold["t"], 2.5, _amp, 0.0, 50.0)
            _hold["t"] += 1
            return a

        run_episode(env, act, seed=0, on_step=on_step)
        env.close()
        if not tr["FL"]:
            print(f"{amp_u:>8.2f} {amp_u * ascale:>9.2f} |  (没走到 {skip} 步就结束了)")
            continue
        cells = [f"{np.max(v):>7.3f}(底{np.max(v) - 0.022:>6.3f})" for v in tr.values()]
        print(f"{amp_u:>8.2f} {amp_u * ascale:>9.2f} | " + " ".join(f"{c:>18}" for c in cells))
    print("\n跨 8 cm 的槛，**脚底**（球心减半径 0.022）至少要高过 0.08。")
    return 0


def cmd_frontier(argv):
    """扫 action_scale × 槛高，量出'动作幅度 -> 能跨多高的槛'。"""
    from go2_common import terrain as T

    def build(h):
        spec = mujoco.MjSpec.from_file(DEFAULT_SCENE)
        for g in [g for g in spec.worldbody.geoms
                  if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.size[2] > T._THRESHOLD_HALF_Z]:
            spec.delete(g)
        for g in spec.worldbody.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.size[2] <= T._THRESHOLD_HALF_Z:
                g.size[2], g.pos[2] = h / 2.0, h / 2.0
        return spec.compile()

    heights = (0.04, 0.06, 0.08, 0.10, 0.12)
    print("开环小跑（无反馈）能跨过的槛高：\n")
    print(f"{'action_scale':>12} | " + " ".join(f"{h:>9.2f}m" for h in heights))
    for a in (0.4, 0.5, 0.6, 0.8):
        cells = []
        for h in heights:
            env = Go2TerrainEnv(EnvCfg(terrain="steps", action_scale=a), RewardCfg())
            env.model = build(h)
            env.data = mujoco.MjData(env.model)
            env.terrain.__init__(env.model)
            x, lv, _, _ = run_once(env, lambda o, e: trot_action(e.step_count, 2.5, 1.0, 0.0, 50.0))
            on_top = env.terrain.top(max(x, 1.2)) > h / 2.0
            cells.append(f"{'过' if on_top else '卡'} x{x:>5.2f}")
            env.close()
        print(f"{a:>12.2f} | " + " ".join(f"{c:>9}" for c in cells))
    print("\n带反馈的策略比这套开环小跑大约好一档。")
    return 0


def cmd_policy(argv):
    path = argv[0]
    terrain, scale = (argv[1], float(argv[2])) if len(argv) > 2 else ("flat", 1.0)
    model = load_policy(path)
    cfg = cfg_from_model(path)
    cfg.terrain, cfg.terrain_scale = terrain, scale
    env = Go2TerrainEnv(cfg, RewardCfg())
    ids, tr = foot_ids(env), {n: [] for n in FOOT_GEOM_NAMES}
    for ep in range(3):
        def on_step(e):
            g = e.terrain.top(float(e.data.qpos[0]), float(e.data.qpos[1]))
            for n, i in ids.items():
                tr[n].append(float(e.data.geom_xpos[i][2]) - g)
        run_episode(env, policy_act(model), seed=300 + ep, on_step=on_step)
    print(f"{path}  地形={terrain} scale={scale}\n")
    for n, v in tr.items():
        v = np.array(v)
        print(f"  {n}: 最高 {v.max():.3f}  中位 {np.median(v):.3f}  "
              f"贴地(<0.03)时间占比 {(v < 0.03).mean() * 100:5.1f}%")
    print("\n跨 8 cm 的槛要抬到 0.102 m。后腿抬不起来 = 前腿上了槛、后腿跟不上，卡成跨坐姿势。")
    env.close()
    return 0


def cmd_why(argv):
    path, scale = argv[0], float(argv[1])
    n_ep = int(argv[2]) if len(argv) > 2 else 6
    model = load_policy(path)
    cfg = cfg_from_model(path)
    cfg.terrain, cfg.terrain_scale = "full", scale
    env = Go2TerrainEnv(cfg, RewardCfg())
    print(f"{path}  scale={scale}  action_scale={cfg.action_scale}  "
          f"起始 x∈{tuple(cfg.reset_x)}  max_episode_s={cfg.max_episode_s}\n")
    for ep in range(n_ep):
        # 自己记而不是用 run_once：要连出生点一起打出来。出生时踩在棱上会立刻侧翻，
        # 光看最远 x 会把它误当成"走不动"（落点机制见 AGENT.md §5-C1）。
        st = {"x": -9.0, "lv": 0}

        def on_step(e):
            p = np.array(e.data.qpos[0:3], dtype=float)
            if "x0" not in st:
                st["x0"], st["z0"] = float(p[0]), float(p[2])
                st["thr0"] = float(e.terrain.top(float(p[0]), float(p[1])))
            st["x"] = max(st["x"], float(p[0]))
            st["lv"] = max(st["lv"], e.terrain.level(float(p[0]), float(p[1])))

        total, info = run_episode(env, policy_act(model), seed=700 + ep, on_step=on_step)
        p = np.array(env.data.qpos[0:3], dtype=float)
        rpy = quat_to_rpy(env.data.qpos[3:7])
        thr = env.terrain.top(p[0], p[1])
        why = []
        if p[2] - thr < cfg.fall_clearance:
            why.append(f"贴地(余量{p[2] - thr:.3f})")
        if abs(rpy[0]) > cfg.flip_rad or abs(rpy[1]) > cfg.flip_rad:
            why.append(f"侧翻(roll{rpy[0]:+.2f} pitch{rpy[1]:+.2f})")
        if abs(p[1]) > cfg.out_y or p[0] < cfg.out_x_back:
            why.append(f"出界(y{p[1]:+.2f})")
        if not why:
            why.append("超时")
        print(f"  回合{ep}: 出生x={st['x0']:+5.2f} 落点地形{st['thr0']:.3f} 落高{st['z0']:.3f} | "
              f"步数{env.step_count:>5} 最远x={st['x']:+6.2f} 回报{total:>8.1f} "
              f"台阶{st['lv']}  -> {' / '.join(why)}")
    env.close()
    return 0


def cmd_reward(argv):
    from go2_common.reward import TERMS

    path, scale = argv[0], float(argv[1])
    n_ep = int(argv[2]) if len(argv) > 2 else 4
    terr = argv[3] if len(argv) > 3 else "full"
    model = load_policy(path)
    cfg = cfg_from_model(path)
    cfg.terrain, cfg.terrain_scale = terr, scale
    env = Go2TerrainEnv(cfg, RewardCfg())
    names = [n for n, _ in TERMS]
    sums = {n: 0.0 for n in names}
    steps = 0
    for ep in range(n_ep):
        _, _, _, info = run_once(env, policy_act(model), seed=500 + ep)
        n = max(env.step_count, 1)
        steps += n
        for k in names:
            sums[k] += float(info[k])
    print(f"{path}  地形={terr} scale={scale}  共 {n_ep} 回合 / {steps} 步  "
          f"(平均 {steps / n_ep:.0f} 步/回合)\n")
    for k, v in sorted(sums.items(), key=lambda kv: kv[1]):
        print(f"  {k:<16} {v:>10.1f}   ({v / steps:+.4f}/步)")
    print("\n注意看回合有多短：如果步数远小于 1000，说明是先摔了，")
    print("各项分自然都接近 0 —— 那不是'奖励设计错了'，是还没走起来就结束了。")
    env.close()
    return 0


COMMANDS = {
    "terrain": cmd_terrain, "stance": cmd_stance, "lift": cmd_lift,
    "frontier": cmd_frontier, "policy": cmd_policy, "why": cmd_why, "reward": cmd_reward,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
