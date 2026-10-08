"""把 `core.py`（Isaac 侧的观测/奖励/终止）和真 `Go2TerrainEnv` 逐位对拍。

    python3 -m go2_issac.tests.core_parity

分两部分，各自管一件事：

1. **合成状态**：直接造一大片"什么姿态都有"的状态（含顶平台、翻倒、出界，以及四条终止判据的
   **边界值**），把观测 / 14 项奖励 / 终止判据的整个状态空间扫一遍。不依赖动力学，
   所以能确定性地覆盖 success / fall / flip / out_of_course——靠 rollout 是撞不齐的。
2. **真 rollout**：用开环小跑（会真的跨槛、上台阶）跑满回合，逐步比。这一段验证的是
   **状态量的映射**：陀螺是不是机体系、线速度是不是世界系、q/dq 是不是执行器顺序、
   力矩取的是不是 actuator_force —— 这些合成状态测不出来。

约定：所有"一批状态"都用同一个 dict 表示，每个字段的形状是 `(n, ...)`，
`to_core_state()` 直接照着包成 `core.State`。观测/奖励的期望值一律用**真实现**算
（`env._obs()` / `reward.compute_reward`），不在测试里重抄一遍公式。

本机没有显卡，所以这是"上机之前唯一能证明 Isaac 侧语义没写错"的办法。
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from go2_common import terrain as mj_terrain  # noqa: E402
from go2_common.config import DEFAULT_SCENE, EnvCfg, RewardCfg, quat_to_rpy  # noqa: E402
from go2_common.env import Go2TerrainEnv  # noqa: E402
from go2_common.reward import RewardInput, compute_reward  # noqa: E402
from go2_issac import core  # noqa: E402

TOL_OBS = 1e-6    # 观测最终转 float32；两边同一套公式，实测差 0
TOL_PART = 1e-9   # 分项奖励（float64）
TOL_TOTAL = 1e-8  # 总和比各项松一点：np.exp / torch.exp 不保证逐位相同

CASES = tuple(
    (variant, scale) for variant in ("flat", "steps", "full") for scale in (0.7, 1.0)
)

_failures: list[str] = []
_max_diff: dict[str, float] = {}


def check(cond: bool, msg: str) -> None:
    if not cond:
        _failures.append(msg)
        if len(_failures) <= 25:
            print(f"  FAIL {msg}")


def note_diff(tag: str, got, want) -> float:
    """记录最大差，并返回它——"差多少"比"过没过"更有诊断价值。"""
    d = float(np.max(np.abs(np.asarray(got, dtype=np.float64)
                            - np.asarray(want, dtype=np.float64))))
    _max_diff[tag] = max(_max_diff.get(tag, 0.0), d)
    return d


def _t(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def pack(f: dict) -> torch.Tensor:
    return _t(f["base_pos"])


# ------------------------------------------------------------------ 打包 / 解包


def to_core_state(f: dict, terrain_h, terrain_level) -> core.State:
    """把 `(n, ...)` 的一批状态包成 `core.State`。字段名与含义见 `core.State`。"""
    return core.State(
        base_pos=_t(f["base_pos"]), base_quat=_t(f["quat"]),
        base_lin_vel=_t(f["base_v"]), base_ang_vel=_t(f["base_w"]),
        q=_t(f["q"]), dq=_t(f["dq"]), tau=_t(f["tau"]),
        contact_force=_t(f["contact_force"]), action=_t(f["action"]),
        terrain_h=_t(terrain_h), terrain_level=_t(terrain_level),
        prev_x=_t(f["prev_x"]), prev_action=_t(f["prev_action"]),
        prev_terrain_h=_t(f["prev_terrain_h"]), prev_terrain_level=_t(f["prev_terrain_level"]),
        cmd_vx=_t(f["cmd_vx"]),
    )


def roll_pitch_yaw_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 欧拉角 -> 四元数 (w, x, y, z)，用来造"恰好翻到阈值"的姿态。"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


# ------------------------------------------------------------------ 合成状态


def synthetic_state(env, n: int, rng: np.random.Generator) -> dict:
    """造一批覆盖整个状态空间的状态。

    * 位置铺满课程（-1.5~4.5 m，含顶平台 x>3.3 和悬崖外 x>3.66），y 铺到出界（|y|>2.5）；
    * 高度 0~1.4 m：低到能触发摔倒、高到在顶平台上正常站立；
    * 姿态一半随机（真的会翻倒）、一半接近竖直；
    * 前 12 个点**刻意**取在四条判据的边界上（含取等号与差 1e-9 两侧），
      盯的是"<= 还是 <""|·| 的边界漏不漏"。
    """
    cfg = env.cfg
    x = rng.uniform(-1.5, 4.5, n)
    y = rng.uniform(-3.0, 3.0, n)
    z = rng.uniform(0.0, 1.4, n)

    quat = rng.normal(size=(n, 4))
    quat[: n // 2] = np.array([1.0, 0.0, 0.0, 0.0]) + rng.normal(scale=0.05, size=(n // 2, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    # --- success / out_of_course 的边界
    x[:5] = [cfg.goal_x, cfg.goal_x, cfg.goal_x, cfg.goal_x - 1e-9, cfg.goal_x + 1e-9]
    y[:5] = [0.0, cfg.goal_y, cfg.goal_y + 1e-9, 0.0, 0.0]
    z[:5] = 1.2
    x[5:7] = [cfg.out_x_back, cfg.out_x_back - 1e-9]  # 倒退出界
    y[5:7] = 0.0
    z[5:7] = 0.5
    x[7:9] = 0.0
    y[7:9] = [cfg.out_y, cfg.out_y + 1e-9]  # 横向出界（等于阈值 / 差一点）
    z[7:9] = 0.3

    # --- 摔倒的边界：z - 支撑面高度 恰好等于 / 差一点小于 fall_clearance
    # z 这里钉不了：支撑面取自**四只脚**，得先把姿态摆进 MuJoCo 才读得到，
    # 由 check_synthetic 补（理由见 terrain.support_height）。
    x[9], y[9] = 3.0, 0.0     # 台阶中段（支撑面高度不为 0，才测得出"相对脚下地面"）
    x[10], y[10] = 2.0, 0.0   # 第 1 级台阶前
    x[11], y[11], z[11] = 3.3, 0.0, 1.2  # 顶平台上正常站立（成功判据该为真）

    # --- 翻转的边界：|roll| 或 |pitch| 恰好等于 flip_rad
    for i, (roll, pitch) in enumerate([(0.0, 0.0), (cfg.flip_rad, 0.0), (-cfg.flip_rad, 0.0),
                                       (0.0, cfg.flip_rad + 1e-9), (0.5, 0.5)]):
        quat[i] = roll_pitch_yaw_to_quat(roll, pitch, 0.0)

    # 上一拍的位置错开一点，让 progress / climb / level_bonus 三项都非零。
    # prev_terrain_* 同理得等四只脚读出来（check_synthetic 里补）。
    prev_x = x - rng.uniform(-0.05, 0.05, n)

    lo, hi = env.q_lo, env.q_hi  # 关节角铺满整个限位范围
    q = rng.uniform(-1.0, 1.0, (n, 12)) * (hi - lo) / 2 + (lo + hi) / 2
    return {
        "base_pos": np.stack([x, y, z], axis=1),
        "quat": quat,
        "base_v": rng.normal(scale=0.5, size=(n, 3)),
        "base_w": rng.normal(scale=0.5, size=(n, 3)),
        "q": q,
        "dq": rng.normal(scale=2.0, size=(n, 12)),
        "tau": rng.normal(scale=10.0, size=(n, 12)),
        "action": rng.uniform(-1.0, 1.0, (n, 12)),
        "prev_action": rng.uniform(-1.0, 1.0, (n, 12)),
        "contact_force": rng.uniform(0.0, 200.0, n),
        "prev_x": prev_x,
        "cmd_vx": rng.uniform(*cfg.cmd_vx, n),
    }


def _pose(env, f: dict, i: int) -> None:
    """把第 i 个合成状态写进 `env.data` 并**前推一次**。

    `mj_forward` 不能省：`geom_xpos` 只在 `mj_step`/`mj_forward` 里更新，只写 qpos 的话
    四只脚还停在上一个状态的位姿上，`env._support_height()` 读到的是别人脚下那片地形。
    """
    env.data.qpos[:3] = f["base_pos"][i]
    env.data.qpos[3:7] = f["quat"][i]
    env.data.qpos[env.qadr] = f["q"][i]
    env.data.qvel[:3] = f["base_v"][i]
    env.data.qvel[3:6] = f["base_w"][i]
    env.data.qvel[env.vadr] = f["dq"][i]
    mujoco.mj_forward(env.model, env.data)


def _feet_xy(env) -> np.ndarray:
    """当前位姿下四只脚的 (x, y)，(脚数, 2)。"""
    return env.data.geom_xpos[env.foot_geom_ids][:, :2].copy()


def _env_obs(env, f: dict, i: int) -> np.ndarray:
    """把第 i 个状态写进 env.data 再调真的 `env._obs()`（不在测试里重抄观测公式）。

    `env.prev_action` 喂的是 **`f["action"]`（本步动作）**：`env.step()` 是先把 `prev_action`
    推进、再构造返回的观测，所以步 t 的观测里装的是 a_t。刻意喂和 `prev_action` 不同的值，
    这样"core 的观测取错了字段"会被立刻抓到（改对之前这里确实红过）。
    """
    _pose(env, f, i)
    env.prev_action = f["action"][i]
    env.cmd_vx = float(f["cmd_vx"][i])
    return env._obs()


# ------------------------------------------------------------------ 三项检查


def check_torch_terrain(env, ct: core.CourseTerrain, tag: str) -> None:
    """torch 地形 vs MuJoCo 地形：top / level / top_footprint 逐位相等。

    采样点特意堆在**方块棱上**（±0 / ±1e-9）——`top_footprint` 的 9 个采样点只有写成
    `np.linspace` 的算法（`start + delta*i`、末点钉成 stop）才会和 `terrain.py` 采在同一批点上，
    否则会出现"某个采样点恰好落在棱上、两边一个算进去一个没算进去"的假阳性。
    """
    edges = np.array([e for b in env.terrain.boxes for e in (b[0], b[1], b[2], b[3])])
    if edges.size:
        ex = np.concatenate([edges, edges + 1e-9, edges - 1e-9])
        X, Y = np.meshgrid(ex, ex)
        x, y = X.ravel(), Y.ravel()
    else:
        x, y = np.array([0.0]), np.array([0.0])
    rng = np.random.default_rng(7)
    n_rand = 1000
    x = np.concatenate([x, rng.uniform(-2.6, 4.6, n_rand)])
    y = np.concatenate([y, rng.uniform(-2.6, 2.6, n_rand)])
    # 棱上的点太多，抽一部分即可（top_footprint 在 Python 里是 27×8 次比较，很慢）
    keep = min(x.size, 3000)
    x, y = x[:keep], y[:keep]

    want = np.array([env.terrain.top(float(a), float(b)) for a, b in zip(x, y)])
    got = ct.top(_t(x), _t(y)).numpy()
    check(note_diff(f"{tag}: 地形top", got, want) == 0.0, f"{tag}: torch 地形 top 与 MuJoCo 不一致")

    want_lv = np.array([env.terrain.level(float(a), float(b)) for a, b in zip(x, y)])
    got_lv = ct.level(_t(x), _t(y)).numpy()
    check(np.array_equal(got_lv, want_lv),
          f"{tag}: torch 地形 level 有 {int((got_lv != want_lv).sum())} 个点不一致")

    want_fp = np.array([env.terrain.top_footprint(float(a), float(b)) for a, b in zip(x, y)])
    got_fp = ct.top_footprint(_t(x), _t(y)).numpy()
    check(note_diff(f"{tag}: 地形top_footprint", got_fp, want_fp) == 0.0,
          f"{tag}: torch 地形 top_footprint 与 MuJoCo 不一致")

    # 支撑面：把采样点四四分组成"四只脚"，两边逐位比。样本里既有全在同一级的，
    # 也有横跨两级/三级台阶的（棱附近的点被刻意采进来了）。
    m = (x.size // 4) * 4
    feet = np.stack([x[:m], y[:m]], axis=1).reshape(-1, 4, 2)
    want_sh = np.array([env.terrain.support_height(g) for g in feet])
    got_sh = ct.support_height(_t(feet)).numpy()
    check(note_diff(f"{tag}: 支撑面高度", got_sh, want_sh) == 0.0,
          f"{tag}: torch 支撑面高度与 MuJoCo 不一致")


def check_synthetic(env, ct: core.CourseTerrain, tag: str, rng: np.random.Generator,
                    n: int = 512) -> None:
    """合成状态上的观测 / 奖励 / 终止判据对拍。"""
    cfg, reward_cfg = env.cfg, env.reward_cfg
    f = synthetic_state(env, n, rng)

    # --- 支撑面：把每个状态摆进 MuJoCo，读**四只脚**的位置，两边吃同一批点。
    # 支撑面是 climb / level_bonus / base_height / is_fallen 的全部地形输入，
    # 所以这里既要"两边同点同值"，也要确认它确实来自脚而不是机身中心。
    feet = np.empty((n, env.foot_geom_ids.size, 2))
    want_th = np.empty(n)
    for i in range(n):
        _pose(env, f, i)
        feet[i] = _feet_xy(env)
        want_th[i] = env._support_height()

    th = ct.support_height(_t(feet)).numpy()
    lv = ct.level_at_height(_t(th)).numpy()
    want_lv = np.array([env.terrain.level_at_height(float(h)) for h in want_th])
    check(note_diff(f"{tag}: 支撑面高度(合成)", th, want_th) == 0.0,
          f"{tag}: 合成状态里 torch 支撑面高度与 MuJoCo 不一致")
    check(np.array_equal(lv, want_lv), f"{tag}: 合成状态里 level_at_height 不一致")

    # 四只脚分散在两级台阶上时（最常见的跨步姿态），中位数必须落在**两级之间**：
    # 这一条盯的是 torch 侧误用 `torch.median`（偶数个样本它取中间偏下那个，正好差半级）。
    straddle = np.array([[[0.0, 0.0], [0.0, 0.0], [2.2, 0.0], [2.2, 0.0]],
                         [[0.0, 0.0], [2.2, 0.0], [2.4, 0.0], [2.4, 0.0]]])
    want_straddle = np.array([env.terrain.support_height(p) for p in straddle])
    got_straddle = ct.support_height(_t(straddle)).numpy()
    check(note_diff(f"{tag}: 跨级中位数", got_straddle, want_straddle) == 0.0,
          f"{tag}: 四只脚横跨台阶时的中位数不一致（torch 侧别用 torch.median）")

    # 补上"摔倒边界"那两行：支撑面高度只取决于脚，所以现在才填得了 z
    f["base_pos"][9, 2] = want_th[9] + cfg.fall_clearance
    f["base_pos"][10, 2] = want_th[10] + cfg.fall_clearance - 1e-9

    # 上一拍的支撑面：把四只脚整体平移到 prev_x（真实上一拍无从合成，
    # 只要两边吃到的是同一份数，climb / level_bonus 的符号就能被覆盖到）
    prev_feet = feet + (f["prev_x"] - f["base_pos"][:, 0])[:, None, None] * np.array([1.0, 0.0])
    f["prev_terrain_h"] = np.array([env.terrain.support_height(p) for p in prev_feet])
    f["prev_terrain_level"] = np.array(
        [env.terrain.level_at_height(float(h)) for h in f["prev_terrain_h"]])

    st = to_core_state(f, th, lv)

    # --- 观测（45 维与 46 维各一遍）
    for privileged in (False, True):
        env.cfg.privileged = privileged
        want = np.stack([_env_obs(env, f, i) for i in range(n)])
        got = core.obs(st, cfg, env.q_default, privileged=privileged).numpy()
        d = note_diff(f"{tag}: obs{'-priv' if privileged else ''}", got, want)
        check(d <= TOL_OBS, f"{tag}: 观测不一致（最大差 {d:.2e}，privileged={privileged}）")
    env.cfg.privileged = False

    # --- 终止判据：四条 + any
    goal_z = core.goal_z_for(cfg, ct)
    check(goal_z == env.cfg.goal_z, f"{tag}: goal_z {goal_z} != env 的 {env.cfg.goal_z}")
    tm = core.terminations(st, cfg, goal_z)
    want_map = {
        "success": np.array([mj_terrain.is_success(cfg, p) for p in f["base_pos"]]),
        "fallen": np.array([mj_terrain.is_fallen(cfg, float(want_th[i]), p)
                            for i, p in enumerate(f["base_pos"])]),
        "flipped": np.array([mj_terrain.is_flipped(cfg, quat_to_rpy(q)) for q in f["quat"]]),
        "out_of_course": np.array([mj_terrain.is_out_of_course(cfg, p) for p in f["base_pos"]]),
    }
    want_map["any"] = (want_map["success"] | want_map["fallen"]
                       | want_map["flipped"] | want_map["out_of_course"])
    for name, want in want_map.items():
        got = tm[name].numpy()
        check(np.array_equal(got, want),
              f"{tag}: 终止判据 {name} 有 {int((got != want).sum())} 个点不一致"
              f"（core 判真 {int(got.sum())} 个，MuJoCo 判真 {int(want.sum())} 个）")

    # --- 奖励：14 项分项 + 总和
    total, parts = core.compute_reward(st, reward_cfg, tm["any"], tm["success"])
    want_total, want_parts = [], []
    for i in range(n):
        r = RewardInput(
            action=f["action"][i], prev_action=f["prev_action"][i], q=f["q"][i], dq=f["dq"][i],
            tau=f["tau"][i], quat=f["quat"][i], base_v=f["base_v"][i], base_w=f["base_w"][i],
            base_pos=f["base_pos"][i], prev_x=float(f["prev_x"][i]),
            terrain_h=float(th[i]), prev_terrain_h=float(f["prev_terrain_h"][i]),
            terrain_level=int(lv[i]), prev_terrain_level=int(f["prev_terrain_level"][i]),
            cmd_vx=float(f["cmd_vx"][i]), contact_force=float(f["contact_force"][i]),
            terminated=bool(tm["any"][i]), success=bool(tm["success"][i]),
        )
        t, p = compute_reward(r, reward_cfg)
        want_total.append(t)
        want_parts.append([p[name] for name in core.TERM_NAMES])
    want_parts = np.asarray(want_parts)
    got_parts = np.stack([parts[name].numpy() for name in core.TERM_NAMES], axis=1)
    d = note_diff(f"{tag}: 奖励总和(合成)", total.numpy(), want_total)
    check(d <= TOL_TOTAL, f"{tag}: 总奖励差 {d:.2e}")
    for k, name in enumerate(core.TERM_NAMES):
        d = note_diff(f"{tag}: 奖励项 {name}", got_parts[:, k], want_parts[:, k])
        check(d <= TOL_PART, f"{tag}: 奖励项 {name} 差 {d:.2e}")


def rollout_fields(env, action, prev: dict) -> dict:
    """把 env 当前的状态收成 `(1, ...)` 的一批字段。"""
    d = env.data
    return {
        "base_pos": d.qpos[:3][None, :].copy(),
        "quat": d.qpos[3:7][None, :].copy(),
        "base_v": d.qvel[0:3][None, :].copy(),
        "base_w": d.qvel[3:6][None, :].copy(),
        "q": d.qpos[env.qadr][None, :].copy(),
        "dq": d.qvel[env.vadr][None, :].copy(),
        "tau": d.actuator_force[None, :].copy(),
        "action": np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)[None, :],
        "contact_force": np.array([env._contact_force()]),
        # 四只脚的课程坐标，(1, 脚数, 2)：支撑面高度取自这里（不是机身中心）
        "foot_xy": _feet_xy(env)[None, :, :],
        "cmd_vx": np.array([env.cmd_vx]),
        "prev_x": np.array([prev["prev_x"]]),
        "prev_action": np.asarray(prev["prev_action"], dtype=np.float64)[None, :].copy(),
        "prev_terrain_h": np.array([prev["prev_terrain_h"]]),
        "prev_terrain_level": np.array([prev["prev_terrain_level"]]),
    }


def check_rollout(env, ct: core.CourseTerrain, tag: str, rng: np.random.Generator,
                  n_episodes: int, max_steps: int) -> dict:
    """开环小跑 / 随机动作各跑几条，逐步对拍；顺带收集"确实走起来了"的统计。"""
    from go2_sac.train import trot_action

    control_hz = 1.0 / (env.cfg.sim_dt * env.cfg.decimation)
    goal_z = core.goal_z_for(env.cfg, ct)
    stats = {"steps": 0, "max_x": -9.0, "max_level": 0, "falls": 0, "successes": 0,
             "levels": []}
    for ep in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 10 ** 6)))
        freq, phase = rng.uniform(2.0, 3.5), rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.24, 0.40) / env.cfg.action_scale  # 绝对摆幅 -> 动作值
        ep_levels, t = set(), 0
        while t < max_steps:
            action = (trot_action(t, freq, amp, phase, control_hz) if ep % 2 == 0
                      else rng.normal(0.0, 0.3, 12).astype(np.float32))
            prev = {"prev_action": env.prev_action.copy(), "prev_x": env.prev_x,
                    "prev_terrain_h": env.prev_terrain_h,
                    "prev_terrain_level": env.prev_terrain_level}
            obs_mj, rew, term, trunc, info = env.step(action)
            f = rollout_fields(env, action, prev)

            # 支撑面：用 core 的 torch 值喂 State，同时和 env 这一步真正用的值比（每一步都盯）。
            # 比的是 `info["terrain_h"]` 而不是现算一遍——那是 step() 内部的实际输入，
            # 顺便把"geom_xpos 没前推、读到上一拍的脚"这类时序错也一起抓了。
            th = ct.support_height(_t(f["foot_xy"])).numpy()
            lv = ct.level_at_height(_t(th)).numpy()
            check(th[0] == info["terrain_h"],
                  f"{tag}: rollout 第 {t} 步支撑面高度不一致"
                  f"（core={th[0]:.9f} env={info['terrain_h']:.9f}）")
            st = to_core_state(f, th, lv)

            got_obs = core.obs(st, env.cfg, env.q_default).numpy()[0]
            d = note_diff(f"{tag}: obs(rollout)", got_obs, obs_mj)
            check(d <= TOL_OBS, f"{tag}: rollout 第 {t} 步观测不一致 {d:.2e}")

            tm = core.terminations(st, env.cfg, goal_z)
            check(bool(tm["any"][0]) == bool(term),
                  f"{tag}: rollout 第 {t} 步 terminated 不一致（core={bool(tm['any'][0])} "
                  f"env={term}，x={f['base_pos'][0, 0]:.3f} z={f['base_pos'][0, 2]:.3f} "
                  f"h={th[0]:.3f}）")
            check(bool(tm["success"][0]) == bool(info["success"]),
                  f"{tag}: rollout 第 {t} 步 success 不一致")
            check(bool(trunc) == bool((not term) and env.step_count >= env.max_steps),
                  f"{tag}: rollout 第 {t} 步 truncated 不一致")

            total, parts = core.compute_reward(st, env.reward_cfg, tm["any"], tm["success"])
            d = note_diff(f"{tag}: 奖励总和(rollout)", total.numpy(), [rew])
            check(d <= TOL_TOTAL, f"{tag}: rollout 第 {t} 步总奖励差 {d:.2e}")
            for name in core.TERM_NAMES:
                # info 里 reward 分项的同名 "success" 被布尔值覆盖了，加权分项存在 success_bonus
                want = info["success_bonus"] if name == "success" else info[name]
                d = note_diff(f"{tag}: 奖励项 {name}(rollout)", parts[name].numpy()[0], want)
                check(d <= TOL_PART, f"{tag}: rollout 第 {t} 步奖励项 {name} 差 {d:.2e}")

            stats["steps"] += 1
            stats["max_x"] = max(stats["max_x"], float(f["base_pos"][0, 0]))
            ep_levels.add(int(lv[0]))
            if info["success"]:
                stats["successes"] += 1
            elif term:
                stats["falls"] += 1
            t += 1
            if term or trunc:
                break
        stats["max_level"] = max(stats["max_level"], max(ep_levels) if ep_levels else 0)
        stats["levels"].append(sorted(ep_levels))
    return stats


# ------------------------------------------------------------------ 主流程


def main() -> int:
    rng = np.random.default_rng(0)
    print(f"对拍 core.py <-> go2_common/env.py + reward.py"
          f"（scene: {pathlib.Path(DEFAULT_SCENE).name}）")
    for variant, scale in CASES:
        env = Go2TerrainEnv(EnvCfg(terrain=variant, terrain_scale=scale), RewardCfg())
        ct = core.CourseTerrain(variant, scale)
        tag = f"{variant}/{scale:g}"
        print(f"\n[{tag}] 台阶 {len(env.terrain.stair_tops)} 级，goal_z={env.cfg.goal_z}，"
              f"max_steps={env.max_steps}，q_default={np.round(env.q_default, 4)}")
        check_torch_terrain(env, ct, tag)
        check_synthetic(env, ct, tag, rng, n=512)
        stats = check_rollout(env, ct, tag, rng, n_episodes=4, max_steps=env.max_steps)
        print(f"  rollout: {stats['steps']} 步，最远 x={stats['max_x']:+.2f}，"
              f"最高台阶 {stats['max_level']}，摔倒 {stats['falls']}，"
              f"成功 {stats['successes']}，每回合到过的级 {stats['levels']}")
        del env, ct

    print("\n最大差（全 0 才是逐位一致）:")
    nonzero = {k: v for k, v in _max_diff.items() if v > 0}
    for k, v in sorted(nonzero.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {k}: {v:.3e}")
    if not nonzero:
        print("  全部为 0")

    print("\n" + ("全部通过" if not _failures else f"{len(_failures)} 项失败:"))
    for f in _failures[:20]:
        print(f"  - {f}")
    if len(_failures) > 20:
        print(f"  ... 还有 {len(_failures) - 20} 项")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
