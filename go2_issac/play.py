"""回放训练好的策略（Isaac Sim 里看狗跑）。**只能在有 NVIDIA 显卡的机器上跑**。

    python3 go2_issac/play.py --resume models/go2_issac_full_20261007_120000
    python3 go2_issac/play.py --resume <那个目录>/model_500.pt --episodes 10 --visualizer none

环境参数（哪档地形、多大的缩放、action_scale、观测维度）**全部从存档同目录的
`config.json` 里还原**，不重新猜——观测维度或奖励权重对不上时，策略不会报错，
只会走得莫名其妙，那种错最难查。命令行只留"看几回合"这种和策略无关的开关。

和 `go2_ppo/play.py` 一样每个回合打印一行（回报 / 最远 x / 最高台阶 / 是否成功），
两边的数不能直接比（物理不同），但"在同一个地形档位上，Isaac 训出来的到底走不走得过去"
是能看的。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from importlib.metadata import version

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="回放 Isaac 里训出来的 Go2 过地形策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--resume", required=True,
                   help="训练输出目录，或直接给 model_<迭代数>.pt")
    p.add_argument("--episodes", type=int, default=5, help="回放几个回合")
    p.add_argument("--num-envs", type=int, default=1,
                   help="并行几个环境（看的话 1 个就够；>1 时打印的是每步结束的回合）")
    p.add_argument("--seed", type=int, default=10_000,
                   help="回放用固定种子（默认和 MuJoCo 那边评估的种子同一批）")
    p.add_argument("--steps", type=int, default=None, metavar="N",
                   help="最多跑多少步（默认按回合长度上限，等于跑满 --episodes 个回合）")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.resume_path = _resolve_resume(args.resume)
    return args


def _resolve_resume(resume: str) -> pathlib.Path:
    """和 `train.py:_resolve_resume` 同一套规则（那边还多一层 sim 检查）。"""
    rp = pathlib.Path(resume)
    if rp.suffix == ".zip":
        raise SystemExit(f"{rp} 是 SB3 的模型，Isaac 这边加载不了（见 RUNBOOK）。")
    if rp.is_dir():
        # 只认纯数字后缀，跳过 `model_interrupted.pt` 这种（理由见 train.py 同函数）
        pt = [(int(q.stem.rsplit("_", 1)[-1]), q) for q in rp.glob("model_*.pt")
              if q.stem.rsplit("_", 1)[-1].isdigit()]
        if not pt:
            raise SystemExit(f"{rp} 里没有 model_<迭代数>.pt")
        return max(pt)[1]
    if not rp.exists() and rp.with_suffix(".pt").exists():
        return rp.with_suffix(".pt")
    if not rp.exists():
        raise SystemExit(f"找不到 {rp}")
    return rp


def _env_cfg_from_run(run_dir: pathlib.Path):
    """从 `config.json` 还原环境配置（键名和 `go2_ppo` 的写法一致）。"""
    from go2_issac import env_cfg as ec

    cfg = ec.CourseEnvCfg()
    f = run_dir / "config.json"
    if not f.exists():
        print(f"  {f} 不在，用 CourseEnvCfg 的默认值（full@1.0）")
        return cfg
    env_part = json.loads(f.read_text(encoding="utf-8")).get("env", {})
    cfg.terrain_variant = env_part.get("terrain", cfg.terrain_variant)
    cfg.terrain_scale = float(env_part.get("terrain_scale", cfg.terrain_scale))
    cfg.action_scale = float(env_part.get("action_scale", cfg.action_scale))
    cfg.privileged = bool(env_part.get("privileged", cfg.privileged))
    print(f"  从 config.json 还原：地形 {cfg.terrain_variant}@{cfg.terrain_scale:g}，"
          f"action_scale {cfg.action_scale:g}，观测 {'46' if cfg.privileged else '45'} 维")
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from go2_issac import env_cfg as ec
    from go2_issac.agents import Go2IsaacPPORunnerCfg
    from go2_issac.mdp import metrics

    cfg = _env_cfg_from_run(args.resume_path.parent)
    cfg.seed = args.seed
    cfg.set_envs(args.num_envs)
    print(cfg.describe())

    agent_cfg = Go2IsaacPPORunnerCfg()
    agent_cfg.device = getattr(args, "device", "cuda:0")
    # 同 `train.py`：rsl-rl 5.x 要先把废弃字段（`stochastic` 等）从配置里摘掉再 `to_dict()`，
    # 否则建 runner 时 `MLPModel` 收到不认识的参数直接 TypeError
    handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    env = ec.build_env(cfg)
    env = RslRlVecEnvWrapper(env)
    # 窗口设 1：`last_episode` 就是刚结束的那一个回合，正好一行一回合
    tracker = metrics.EpisodeTracker(env, num_levels=cfg.num_levels(), window=1, verbose=False)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.resume_path))
    policy = runner.get_inference_policy(device=agent_cfg.device)
    print(f"  已加载 {args.resume_path}")

    # 回合上限 = 1000 步（20 s @50 Hz）；多跑一点余量，免得刚好卡在边界上
    # （3.0 起 `RslRlVecEnvWrapper` 不再透传未知属性，2.3 时能直接 `env.step_dt`；
    #   `step_dt` 在底下的 `ManagerBasedRLEnv` 上，走 `unwrapped` 拿）
    limit = args.steps or (args.episodes * (int(cfg.episode_length_s / env.unwrapped.step_dt) + 2))
    n_done, seen, run_steps = 0, 0, 0
    obs, _ = _reset(env)
    try:
        for run_steps in range(1, limit + 1):
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = tracker.step(actions)
            if tracker.episodes != seen:
                seen = tracker.episodes
                ep = tracker.last_episode
                n_done += 1
                print(f"  回合 {n_done}（{ep['steps']:>4} 步）  "
                      f"回报 {ep['reward']:8.1f}  "
                      f"最远 x {ep['max_x']:+.2f}  "
                      f"速度 {ep['vx']:+.2f}/{ep['cmd']:.2f}  "
                      f"最高台阶 {ep['level']}/{cfg.num_levels()}  "
                      f"{'成功' if ep['success'] else '未成功'}")
                if n_done >= args.episodes:
                    break
    except KeyboardInterrupt:
        print("\n  手动中断")
    finally:
        if n_done < args.episodes:
            print(f"  跑了 {run_steps} 步，只结束了 {n_done} 个回合（没跑满 {args.episodes} 个）")
        env.close()
        simulation_app.close()
    return 0


def _reset(env):
    """`env.reset()` 的返回值在 rsl_rl 1.x/2.x 之间变过（2 元组 vs 1 个 obs）。"""
    out = env.reset()
    return out if isinstance(out, tuple) else (out, {})


if __name__ == "__main__":
    sys.exit(main())
