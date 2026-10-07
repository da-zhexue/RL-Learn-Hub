"""用 rsl_rl 的 PPO 在 Isaac Sim 里训练 Go2 过地形。**只能在有 NVIDIA 显卡的机器上跑**
（见 `RUNBOOK.md`；本机连 Isaac Sim 都装不下）。

    python3 go2_issac/train.py --terrain flat  --steps 200_000 --headless
    python3 go2_issac/train.py --terrain steps --steps 100_000 --num-envs 512 --headless
    python3 go2_issac/train.py --terrain full  --steps 150_000 --num-envs 2048 --headless

课程阶段和 MuJoCo 侧（`go2_ppo/train.py`）完全一样，先易后难：
flat（跑稳）→ steps（过 0.08 m 的槛）→ full（爬台阶）。
输出 `models/go2_issac_<地形>[_s<缩放>]_<时间戳>/`，和那边同一套目录约定 + `config.json`。

**和 `go2_ppo/train.py` 的三处结构性不同**（不是随手改的，理由都在下面标了）：

1. **这里是 GPU 全并行**：几百上千个环境同时跑，"环境步/秒"比 MuJoCo 侧高两个数量级，
   所以 `--steps` 的默认量和 `--num-envs` 的取值范围都不一样。没给默认步数，必须显式指定。
2. **没有 `--n-steps/--batch-size/--n-epochs/...` 这一串**：那些是 PPO 超参，
   全部集中在 `agents/rsl_rl_ppo_cfg.py`（一张和 `go2_ppo/config.py` 的逐项映射表），
   训练脚本只负责"跑哪档地形、跑多久、多少个环境"。
3. **没有独立的 eval 环境**：Isaac 里再起一个物理场景很贵。改成在训练环境里
   **顺带统计**（`mdp/metrics.py:EpisodeTracker`），指标口径和 MuJoCo 侧的
   `EvalMetricsCallback` 逐项对齐，TensorBoard 上可以直接对看。
   `eval/` 那几个数因此是"训练用的随机策略"的统计，不是确定性策略的——
   趋势可用，绝对值不要跨实现比。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# 支持 `python3 go2_issac/train.py` 直接运行（光靠 `python3 -m` 不够，见 go2_ppo/train.py）
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO))

# `go2_issac/__init__.py` 是纯 docstring，`course.py` 是纯 numpy，起 Kit 之前 import 没问题
# （要拿 TERRAINS 做 argparse 的 choices，必须在解析参数之前拿到）
from go2_issac import course  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="用 rsl_rl 的 PPO 在 Isaac Sim 里训练 Go2 过地形（平地 -> 槛 -> 台阶）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--terrain", choices=course.TERRAINS, default="full",
                   help="课程阶段 flat/steps/full（默认 full）")
    p.add_argument("--terrain-scale", type=float, default=1.0, metavar="S",
                   help="地形高度整体缩放（课程用）：0.3 / 0.6 / 1.0 逐级长高，1.0=原场景。"
                        "x/y 脚印不变，只压矮高度。语义同 go2_ppo 那边的同名参数")
    p.add_argument("--action-scale", type=float, default=None, metavar="A",
                   help="关节目标幅度（rad），决定能跨多高的槛。默认沿用 go2_common 的值；"
                        "**续训时要么显式给、要么确认和上次一致**（见 AGENT.md §5-A2）")
    p.add_argument("--steps", type=int, required=True,
                   help="转移数（各环境求和，和 go2_ppo 的 --steps 同口径）：从零训是总量，"
                        "--resume 时是**本级再练多少**（rsl_rl 的 learn() 是在已加载的迭代数上"
                        "往上加，和 go2_ppo 那边'续训是本阶段再练多少步'一致）。"
                        "迭代数 = steps / (num_envs × num_steps_per_env)")
    p.add_argument("--num-envs", type=int, default=512,
                   help="并行环境数。GPU 上一个环境占 ~几 MB 显存，512 起步；"
                        "**地形显存随环境数平方增长**（见 env_cfg._patch_grid 的算术）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None,
                   help="默认 models/go2_issac_<地形>[_s<缩放>]_<时间戳>/")
    p.add_argument("--resume", default=None,
                   help="从 rsl_rl 的 .pt 接着训（**同一个仿真器内**；MuJoCo 的 .zip 不行，"
                        "见 RUNBOOK「跨仿真器不能续训」）。给目录会给最新的 model_*.pt")
    p.add_argument("--privileged", action="store_true",
                   help="观测里加 1 维机身离地高度（真机上没有，仅作对照）")
    p.add_argument("--mini-batches", type=int, default=None, metavar="N",
                   help="PPO 每个 epoch 切成几个 minibatch。默认按 SB3 口径换算 "
                        "n_steps×n_envs/batch_size（即每个 minibatch 恰好 256 条），"
                        "环境一多这个数会很大，嫌迭代慢就往下调")
    p.add_argument("--max-iterations", type=int, default=None, metavar="N",
                   help="直接给迭代数（给了就忽略 --steps）")
    p.add_argument("--torch-threads", type=int, default=None, help="限制 CPU 线程数")

    # --headless / --device / --enable_cameras / --cpu 等由 Kit 自己加
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    # 续训路径的检查放在这里（**起 Kit 之前**）：起一次 Kit 要几十秒，
    # 路径写错、给了 MuJoCo 的 .zip、给了别的仿真器的存档，都不该等那么久才报
    args.resume_path = _resolve_resume(args.resume)
    return args


def _resolve_resume(resume: str | None) -> pathlib.Path | None:
    """校验并补全 `--resume`。纯 pathlib/json，不需要 Kit（所以能提前跑）。"""
    if not resume:
        return None
    rp = pathlib.Path(resume)
    if rp.suffix == ".zip":
        raise SystemExit(
            f"{rp} 是 stable-baselines3 的模型。跨仿真器不能续训（见 RUNBOOK）："
            f"MuJoCo 和 PhysX 的接触模型不一样，权重接过去只会得到一个已经坏掉的策略。"
            f"要在 Isaac 里训就从头训；要接着 Isaac 的训就给 rsl_rl 的 .pt。")
    if rp.is_dir():
        rp = _latest_checkpoint(rp)
    elif rp.suffix != ".pt":
        # rsl_rl 存的是 model_<迭代数>.pt；没写后缀就按这个补
        cand = rp.with_suffix(".pt")
        rp = cand if cand.exists() else rp
    if not rp.exists():
        raise SystemExit(f"找不到 {rp}（rsl_rl 的存档是 model_<迭代数>.pt）")
    cfg_json = _load_run_config(rp.parent)
    if cfg_json and cfg_json.get("sim") not in (None, "isaac"):
        raise SystemExit(f"{rp.parent}/config.json 里 sim={cfg_json.get('sim')!r}，"
                         f"不是 Isaac 训出来的存档")
    return rp


def _latest_checkpoint(path: pathlib.Path) -> pathlib.Path:
    """目录 -> 里面迭代数最大的 `model_<n>.pt`（rsl_rl 每 `save_interval` 存一个）。

    **只认纯数字后缀**：`model_interrupted.pt`（Ctrl-C 那份）和别的非数字名字要跳过，
    否则 `int("interrupted")` 直接 ValueError——而它恰恰是"上次没跑完"时目录里最显眼的那个。
    """
    pt = []
    for q in path.glob("model_*.pt"):
        tail = q.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            pt.append((int(tail), q))
    if not pt:
        raise SystemExit(f"{path} 里没有 model_<迭代数>.pt")
    return max(pt)[1]


def _load_run_config(path: pathlib.Path) -> dict:
    """读 `config.json`（和 go2_ppo 同一套格式）。

    自己读而不是用 `go2_common.train_utils.load_resume_configs`：那个模块顶层就
    `import mujoco` + `stable_baselines3`，Isaac 这台机器上没有（也不该装）。
    """
    f = path / "config.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---------------------------------------------------------- 起 Kit
    # 顺序是硬要求：AppLauncher 必须在**任何** isaaclab/isaacsim 模块之前跑起来，
    # 否则 Kit 会以"没配好"的状态初始化，报的错和真正的原因差很远。
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 下面这些 import 都要求 Kit 已经起来，**不要**提到上面去
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from go2_common.config import MODELS_DIR, RewardCfg, to_jsonable
    from go2_issac import env_cfg as ec
    from go2_issac.agents import Go2IsaacPPORunnerCfg, iterations_for, mini_batches
    from go2_issac.mdp import metrics

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    # ---------------------------------------------------------- 配置
    cfg = ec.CourseEnvCfg()
    cfg.terrain_variant = args.terrain
    cfg.terrain_scale = args.terrain_scale
    cfg.privileged = bool(args.privileged)
    if args.action_scale is not None:
        cfg.action_scale = args.action_scale
    cfg.seed = args.seed
    cfg.set_envs(args.num_envs)

    agent_cfg = Go2IsaacPPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = getattr(args, "device", "cuda:0")
    agent_cfg.algorithm.num_mini_batches = (
        args.mini_batches if args.mini_batches is not None
        else mini_batches(args.num_envs)
    )
    iters = args.max_iterations or iterations_for(
        args.steps, args.num_envs, agent_cfg.num_steps_per_env)
    agent_cfg.max_iterations = iters

    # 续训路径已在 `parse_args` 里校验完（起 Kit 之前就报错，见 `_resolve_resume`）
    resume_path = args.resume_path
    if resume_path is not None:
        print(f"  续训自 {resume_path}")

    # ---------------------------------------------------------- 落盘
    scale_tag = "" if cfg.terrain_scale == 1.0 else f"_s{cfg.terrain_scale:g}"
    run_dir = (
        pathlib.Path(args.save_dir) if args.save_dir
        else MODELS_DIR / f"go2_issac_{cfg.terrain_variant}{scale_tag}_"
                          f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # `config.json` 沿用 go2_ppo 的 {"env","reward","train"} 三段式，`play.py` 靠它还原因境。
    # 多出来的 "sim"/"iterations"/"mini_batches" 是 Isaac 侧才有的量
    train_meta = {
        "steps": args.steps,
        "iterations": iters,
        "num_envs": args.num_envs,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
        "samples_per_mini_batch": (
            agent_cfg.num_steps_per_env * args.num_envs
            // max(1, agent_cfg.algorithm.num_mini_batches)),
        "seed": args.seed,
    }
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "sim": "isaac",
                "env": to_jsonable(cfg.to_go2_env_cfg()),
                "reward": to_jsonable(RewardCfg()),
                "train": train_meta,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(cfg.describe())
    print(f"  跑 {iters} 轮，每轮 {agent_cfg.num_steps_per_env} 步 × {args.num_envs} 个环境"
          f" = {agent_cfg.num_steps_per_env * args.num_envs:,} 条转移；"
          f"minibatch {agent_cfg.algorithm.num_mini_batches} 个"
          f"（每个约 {train_meta['samples_per_mini_batch']:,} 条）")
    print(f"  输出 {run_dir}")

    # ---------------------------------------------------------- 建环境
    env = ec.build_env(cfg)
    env = RslRlVecEnvWrapper(env)
    # 指标包在**最外面**：它的 `step()` 必须看到 rsl_rl 口径的 4 元组，
    # 而它内部要读的又是底下 ManagerBasedRLEnv 的状态（`__getattr__` 透传）
    env = metrics.EpisodeTracker(env, num_levels=cfg.num_levels())

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir),
                            device=agent_cfg.device)
    if resume_path is not None:
        runner.load(str(resume_path))

    try:
        runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        # 打断也要留个能接着训的存档，不然几小时白跑
        print("\n  收到 Ctrl-C，存一份再退出")
        runner.save(str(run_dir / "model_interrupted.pt"))
    finally:
        env.close()
        simulation_app.close()

    print(f"\n完成。模型在 {run_dir}（`play.py --resume {run_dir}` 回放）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
