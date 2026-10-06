"""回放训练好的 Go2 策略（PPO 版）。

两种模式：
  --mode viewer （默认）在本机开个 MuJoCo 窗口回放，不需要 DDS
  --mode dds    用 ctrl/go2_ctrl.py 把同一套策略通过 rt/lowcmd 下发到 unitree_mujoco
                仿真，将来可以直接换到真机。需要先另开一个终端跑仿真：
                    cd unitree_mujoco/simulate_python && python3 unitree_mujoco.py

用法：
    python3 -m go2_ppo.play --mode viewer --model models/go2_ppo_full_xxx/model.zip
    python3 -m go2_ppo.play --mode dds    --model models/go2_ppo_full_xxx/model.zip

观测定义与训练时完全一致（45 维，全部能从 DDS 的 LowState 复现），
所以两种模式跑的是同一个策略、同一套输入。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from go2_common.config import (  # noqa: E402
    CTRL_DIR,
    DEFAULT_Q,
    DEFAULT_SCENE,
    MODELS_DIR,
    OBS_CLIP,
    OBS_CMD_SCALE,
    OBS_DQ_SCALE,
    OBS_GYRO_SCALE,
    EnvCfg,
    RewardCfg,
    projected_gravity,
    quat_to_rpy,
)
from go2_common.env import Go2TerrainEnv, run_episode  # noqa: E402
from go2_common.terrain import TERRAINS  # noqa: E402
from go2_common.train_utils import CONFIG_NAME, load_resume_configs  # noqa: E402

FALLEN_UP = 0.5  # 机体系"上"方向的 z 分量小于这个值就算翻倒了（DDS 模式用）


def find_default_model() -> pathlib.Path | None:
    """没给 --model 时，挑 models/ 里最新的一个 go2_ppo 的 model.zip。"""
    if not MODELS_DIR.exists():
        return None
    candidates = sorted(MODELS_DIR.glob("go2_ppo_*/model.zip"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_policy(model_path: pathlib.Path):
    """加载策略，并尽量用它训练时的环境配置。"""
    from stable_baselines3 import PPO

    env_cfg, reward_cfg = load_resume_configs(model_path)
    if env_cfg is None:
        env_cfg, reward_cfg = EnvCfg(scene=DEFAULT_SCENE, terrain="full"), RewardCfg()
    model = PPO.load(str(model_path), device="cpu")
    return model, env_cfg, reward_cfg


# ------------------------------------------------------------------ viewer 模式


def play_viewer(model, env_cfg, episodes, verbose=True):
    """在 MuJoCo 窗口里按真实时间回放。"""
    import mujoco.viewer

    env = Go2TerrainEnv(env_cfg)
    period = env.cfg.sim_dt * env.cfg.decimation  # 每个策略步推进的仿真时间 = 真实时间

    def act(obs, _env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(episodes):
            stats = {"max_x": -9.0, "max_level": 0}
            deadline = time.perf_counter()

            def on_step(e):
                nonlocal deadline
                stats["max_x"] = max(stats["max_x"], float(e.data.qpos[0]))
                stats["max_level"] = max(
                    stats["max_level"], e.terrain.level(e.data.qpos[0], e.data.qpos[1])
                )
                deadline += period
                left = deadline - time.perf_counter()
                if left > 0:
                    time.sleep(left)
                viewer.sync()

            total, info = run_episode(env, act, seed=None, on_step=on_step)
            if not viewer.is_running():
                break
            if verbose:
                print(
                    f"  回合 {ep + 1}: 回报 {total:8.1f}  结束 x={info['x']:+.2f}  "
                    f"最远 x={stats['max_x']:+.2f}  最高台阶 {stats['max_level']}"
                    f"/{len(env.terrain.stair_tops)}  "
                    f"{'成功' if info['success'] else '未成功'}"
                )

    # 绕过 glfwTerminate() 的段错误：主动按正常退出码结束进程（见 AGENT.md §5-E5）。
    # 上面的 print 都已经 flush，不会丢输出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ------------------------------------------------------------------ dds 模式


def build_obs_from_state(q, dq, quat, gyro, cmd_vx, prev_action, q_default):
    """按训练时的定义组观测。只有 q/dq/quat/gyro 四样，真机上一样能拿到。"""
    obs = np.concatenate(
        [
            np.asarray(gyro, dtype=np.float64) * OBS_GYRO_SCALE,
            projected_gravity(quat),
            np.array([cmd_vx, 0.0, 0.0]) * OBS_CMD_SCALE,
            np.asarray(q, dtype=np.float64) - q_default,
            np.asarray(dq, dtype=np.float64) * OBS_DQ_SCALE,
            prev_action,
        ]
    )
    return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)


def play_dds(model, env_cfg, args):
    """把策略通过 DDS 下发到 unitree_mujoco 仿真（或真机）。

    姿态一律从四元数自己算，不读 rpy / base 高度 / foot_force（见 AGENT.md §4.4）。
    """
    sys.path.insert(0, str(CTRL_DIR))
    from go2_ctrl import Q_MAX, Q_MIN, Go2Ctrl

    period = env_cfg.sim_dt * env_cfg.decimation
    q_default = np.array(DEFAULT_Q, dtype=np.float64)

    ctrl = Go2Ctrl(kp=env_cfg.kp, kd=env_cfg.kd, dt=env_cfg.sim_dt,
                   domain_id=args.domain_id, interface=args.interface)
    print("等待 rt/lowstate …")
    ctrl.read_state()
    print("回到默认站姿 …")
    ctrl.reset()
    time.sleep(0.5)

    # 有 rt/sportmodestate 的话顺带订阅一下，只用来打印位置（位置不在 LowState 里）
    pose = {}
    try:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sub.Init(lambda m: pose.update(p=np.array(m.position, dtype=float)), 10)
    except Exception as exc:  # 真机上没有这个 topic 也不影响跑策略
        print(f"  (没订阅到 rt/sportmodestate，只是少打印位置：{exc})")

    prev_action = np.zeros(12, dtype=np.float64)
    cmd_vx = float(np.mean(env_cfg.cmd_vx))
    print(f"前进指令 vx = {cmd_vx:.2f} m/s，Ctrl-C 结束")

    for ep in range(args.episodes):
        ctrl.reset()
        prev_action[:] = 0.0
        deadline = time.perf_counter()
        t_start = time.perf_counter()
        n_tilted = 0
        while time.perf_counter() - t_start < args.duration:
            s = ctrl.read_state()
            obs = build_obs_from_state(s["q"], s["dq"], s["quat"], s["gyro"],
                                       cmd_vx, prev_action, q_default)
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float64)

            q_target = np.clip(q_default + env_cfg.action_scale * action, Q_MIN, Q_MAX)
            ctrl.send_cmd(q_target)
            prev_action = action

            # 翻倒就提前结束这一回合（同样只看四元数）
            n_tilted = n_tilted + 1 if -projected_gravity(s["quat"])[2] < FALLEN_UP else 0
            if n_tilted >= 10:
                print("  翻倒了，本回合提前结束")
                break

            deadline += period
            left = deadline - time.perf_counter()
            if left > 0:
                time.sleep(left)

        pos = pose.get("p")
        where = f"  x={pos[0]:+.2f} y={pos[1]:+.2f} z={pos[2]:.2f}" if pos is not None else ""
        rpy = quat_to_rpy(ctrl.read_state()["quat"])
        print(f"  回合 {ep + 1} 结束{where}  roll={rpy[0]:+.2f} pitch={rpy[1]:+.2f} yaw={rpy[2]:+.2f}")

    print("回到默认站姿（保持 2 s）…")
    ctrl.reset()
    time.sleep(2.0)
    ctrl.stop()


# ------------------------------------------------------------------ 辅助


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="回放训练好的 Go2 策略（viewer 本地回放 / dds 下发到仿真或真机）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=None,
                   help="model.zip 路径，默认取 models/go2_ppo_* 下最新的")
    p.add_argument("--mode", choices=("viewer", "dds"), default="viewer")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--duration", type=float, default=20.0, help="dds 模式每个回合的时长，秒")
    p.add_argument("--terrain", choices=TERRAINS, default=None,
                   help="只在模型目录里没有 config.json 时生效")
    p.add_argument("--domain-id", type=int, default=1, help="dds 模式用，默认 1")
    p.add_argument("--interface", default="lo", help="dds 模式用，默认 lo；真机填网卡名")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_path = pathlib.Path(args.model) if args.model else find_default_model()
    if model_path is None or not model_path.exists():
        print("找不到模型：用 --model 指定 model.zip 路径（或者先用 go2_ppo.train 训一个）")
        return 1
    print(f"加载模型 {model_path}")

    model, env_cfg, _reward_cfg = load_policy(model_path)
    # 模型目录里没有 config.json 时，才允许用命令行指定的地形覆盖
    if args.terrain and not (model_path.parent / CONFIG_NAME).exists():
        env_cfg.terrain = args.terrain

    if args.mode == "viewer":
        play_viewer(model, env_cfg, args.episodes)
    else:
        play_dds(model, env_cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
