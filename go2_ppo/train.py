"""用 PPO 训练 Go2 过地形。

用法（在仓库根目录，两种写法都行）：
    python3 -m go2_ppo.train --terrain full --steps 3_000_000
    python3 go2_ppo/train.py --terrain flat --steps 1_000_000 --n-envs 8

建议的课程与 go2_sac 完全相同（先易后难，比重头硬啃台阶快很多）：
    1) --terrain flat  训练到能稳定小跑不摔
    2) --terrain steps 微调，学会抬腿过 0.08 m 的槛
    3) --terrain full  微调爬台阶
    4) --resume 把上一阶段的 model.zip 接着训

训练速度以运行时打印的 fps 为准：PPO 每采一轮（n_steps × n_envs 步）就做 n_epochs 轮更新，
更新/数据的比例和 SAC 完全不同，不能拿 SAC 那个"约 440 环境步/秒"来估。
"""

from __future__ import annotations

import os

# 必须在 import torch 之前设置：每个子进程只跑物理，给它 1 个线程就够，
# 否则 N 个环境进程各开一堆 OMP 线程，互相抢核反而更慢。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

# 支持 `python3 go2_ppo/train.py` 这种直接运行（此前踩过 "No module named 'ctrl'" 的坑）
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.utils import FloatSchedule  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from go2_common.config import (  # noqa: E402
    DEFAULT_SCENE,
    MODELS_DIR,
    RUNS_DIR,
    EnvCfg,
    RewardCfg,
    to_jsonable,
)
from go2_common.terrain import TERRAINS  # noqa: E402
from go2_common.train_utils import (  # noqa: E402
    CONFIG_NAME,
    EvalMetricsCallback,
    load_resume_configs,
    make_env,
    resolve_model_path,
)
from go2_ppo.config import TrainCfg  # noqa: E402


def _lr_or_clip_value(v) -> float:
    """读出 learning_rate / clip_range 的**当前数值**。

    这两个从 `PPO.load()` 回来时可能已经是 `FloatSchedule`（clip_range 一定是，
    `PPO._setup_model` 会无条件包一层），也可能是裸 float（learning_rate 是，
    `BaseAlgorithm.__init__` 只是把它存下来）。所以先判断能不能调用再取值。
    """
    return float(v(1.0)) if callable(v) else float(v)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="用 PPO 训练 Go2 过地形（平地 -> 槛 -> 台阶）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 这几个默认 None = "没显式指定"，这样 --resume 时才能让被续训模型的 config.json 说了算
    p.add_argument("--terrain", choices=TERRAINS, default=None,
                   help="课程阶段 flat/steps/full；默认不指定时用 full（--resume 则沿用模型自己的）")
    p.add_argument("--scene", default=None, help="场景 xml，默认 unitree_mujoco 自带的 scene.xml")
    p.add_argument("--steps", type=int, default=TrainCfg.total_steps, help="总训练步数")
    p.add_argument("--n-envs", type=int, default=TrainCfg.n_envs,
                   help="并行环境数。PPO 一轮 rollout = n_steps × n_envs 步，env 越多每轮样本越多、"
                        "但单个环境的数据越旧；8 是这套时序下调好的默认")
    p.add_argument("--seed", type=int, default=TrainCfg.seed)
    p.add_argument("--save-dir", default=None, help="默认 models/go2_ppo_<地形>_<时间戳>/")
    p.add_argument("--resume", default=None,
                   help="从已有的模型接着训，.zip 可省略；会读它同目录的 config.json（也会往上找一层，"
                        "所以 checkpoints/rl_xxx 这种路径直接给就行）。**只能续 PPO 自己的模型**")
    p.add_argument("--privileged", action="store_true", default=None,
                   help="观测里加 1 维机身离地高度（真机上没有，仅作对照）")
    p.add_argument("--reset-x", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="起点 x 随机范围，例如 --reset-x 1.0 2.0 直接练爬台阶")
    p.add_argument("--terrain-scale", type=float, default=None, metavar="S",
                   help="地形高度整体缩放（课程用）：0.3 / 0.6 / 1.0 逐级长高，1.0=原场景，0=平地。"
                        "x/y 脚印不变，只压矮高度")
    p.add_argument("--action-scale", type=float, default=None, metavar="A",
                   help="关节目标幅度，决定能跨多高的槛。续训时必须显式给，否则沿用 checkpoint 的值（见 AGENT.md §5-A2）")
    p.add_argument("--learning-rate", type=float, default=None, metavar="LR",
                   help="微调时调小（如 1e-4）。续训时不显式给会被 checkpoint 的值静默覆盖（见 AGENT.md §5-A1）")
    p.add_argument("--n-steps", type=int, default=None, metavar="N",
                   help="每个环境每轮 rollout 采多少步（默认 256）。**别用 SB3 默认的 2048**："
                        "50 Hz 下那是 41 s，比整个回合（20 s）还长（见 TrainCfg.n_steps）")
    p.add_argument("--batch-size", type=int, default=None, metavar="B",
                   help="minibatch 大小（默认 256）。建议取 n_steps × n_envs 的约数，否则最后一批是零头")
    p.add_argument("--n-epochs", type=int, default=None, metavar="E",
                   help="每轮 rollout 的数据重复训几遍（默认 5）")
    p.add_argument("--ent-coef", type=float, default=None, metavar="C",
                   help="熵奖励系数（默认 0.005）。PPO 初始 σ=1.0，熵不是主旋钮")
    p.add_argument("--target-kl", type=float, default=None, metavar="KL",
                   help="早停阀（默认 0.02）：单次更新的 approx_kl 超过 1.5×它，本轮剩下的 epoch 全停")
    p.add_argument("--fresh-reward", action="store_true",
                   help="续训时**不**沿用 checkpoint 里的奖励权重，改用当前 RewardCfg 的默认值；"
                        "换课程阶段必须加（见 AGENT.md §5-A2）")
    p.add_argument("--eval-freq", type=int, default=None, metavar="N",
                   help="每 N 个环境步评估一次（默认 50000）")
    p.add_argument("--eval-episodes", type=int, default=None, metavar="N",
                   help="每次评估跑几个回合。**默认 20，别往小调**（理由见 AGENT.md §5-D2）")
    p.add_argument("--checkpoint-freq", type=int, default=None, metavar="N",
                   help="每 N 个环境步存一个 checkpoint（默认 200000）")
    p.add_argument("--torch-threads", type=int, default=TrainCfg.torch_threads)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)

    # 先确定环境：--resume 时以模型的 config.json 为准，命令行显式给的参数再覆盖它。
    # 顺序不能反——反过来会出现"想练台阶但环境还是平地"这种静默错误。
    # PPO 是 on-policy，没有 replay buffer，只要模型那半边
    resume_path, _ = (None, None) if not args.resume else resolve_model_path(args.resume)
    if resume_path is not None:
        env_cfg, reward_cfg = load_resume_configs(resume_path)
        if env_cfg is None:
            env_cfg, reward_cfg = EnvCfg(), RewardCfg()
        if args.fresh_reward:
            reward_cfg = RewardCfg()
            print("  --fresh-reward：奖励权重用当前默认值，不沿用 checkpoint 里的")
        if args.terrain is not None:
            env_cfg.terrain = args.terrain
        if args.scene is not None:
            env_cfg.scene = args.scene
        if args.privileged is not None:
            env_cfg.privileged = bool(args.privileged)
        if args.reset_x is not None:
            env_cfg.reset_x = tuple(args.reset_x)
        if args.terrain_scale is not None:
            env_cfg.terrain_scale = args.terrain_scale
        if args.action_scale is not None:
            env_cfg.action_scale = args.action_scale
        print(f"  续训自 {resume_path}")
    else:
        env_cfg = EnvCfg(
            scene=args.scene or DEFAULT_SCENE,
            terrain=args.terrain or "full",
            privileged=bool(args.privileged),
        )
        if args.reset_x is not None:
            env_cfg.reset_x = tuple(args.reset_x)
        if args.terrain_scale is not None:
            env_cfg.terrain_scale = args.terrain_scale
        if args.action_scale is not None:
            env_cfg.action_scale = args.action_scale
        reward_cfg = RewardCfg()

    # 课程缩放的档位写进目录名，免得同一个地形下 s0.6 和 s1 的模型混在一起分不清
    scale_tag = "" if env_cfg.terrain_scale == 1.0 else f"_s{env_cfg.terrain_scale:g}"
    run_dir = (
        pathlib.Path(args.save_dir)
        if args.save_dir
        else MODELS_DIR / f"go2_ppo_{env_cfg.terrain}{scale_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainCfg(total_steps=args.steps, n_envs=args.n_envs, seed=args.seed)
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.n_steps is not None:
        train_cfg.n_steps = args.n_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.n_epochs is not None:
        train_cfg.n_epochs = args.n_epochs
    if args.ent_coef is not None:
        train_cfg.ent_coef = args.ent_coef
    if args.target_kl is not None:
        train_cfg.target_kl = args.target_kl
    if args.eval_freq is not None:
        train_cfg.eval_freq = args.eval_freq
    if args.eval_episodes is not None:
        train_cfg.eval_episodes = args.eval_episodes
    if args.checkpoint_freq is not None:
        train_cfg.checkpoint_freq = args.checkpoint_freq

    # 把配置落盘：play.py 靠它还原环境，避免观测维度和奖励权重对不上
    with open(run_dir / CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(
            {"env": to_jsonable(env_cfg), "reward": to_jsonable(reward_cfg), "train": to_jsonable(train_cfg)},
            f, indent=2, ensure_ascii=False,
        )
    rollout_batch = train_cfg.n_steps * args.n_envs
    print(f"地形={env_cfg.terrain}(scale={env_cfg.terrain_scale:g})  action_scale={env_cfg.action_scale:g}  "
          f"观测维度={45 + (1 if env_cfg.privileged else 0)}  "
          f"环境数={args.n_envs}  总步数={args.steps:,}  输出目录={run_dir}")
    print(f"每轮 rollout：{train_cfg.n_steps} 步 × {args.n_envs} 环境 = {rollout_batch:,} 条样本，"
          f"{train_cfg.n_epochs} 个 epoch × {rollout_batch // train_cfg.batch_size} 个 minibatch")

    # 每个 --save-dir 一个 tb 子目录（全写进 runs/ 会把不同 run 的曲线混成一条，见 AGENT.md §5-A5）
    tb_dir = RUNS_DIR / run_dir.name
    tb_dir.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv(
        [partial(make_env, env_cfg, reward_cfg, args.seed + i) for i in range(args.n_envs)]
    )

    if resume_path is not None:
        model = PPO.load(str(resume_path), env=vec_env, device="cpu", tensorboard_log=str(tb_dir))
        # 续训必须显式把超参再盖回去：PPO.load() 不传 kwargs，checkpoint 里存的超参
        # 会整个恢复、静默压掉命令行给的（同 SAC，见 AGENT.md §5-A1）。
        #
        # **注意别去调 model._setup_model()。** OnPolicyAlgorithm._setup_model() 里有一句
        # 无条件执行的 `self.policy = self.policy_class(...)`，会**重建策略网络**，
        # 把刚 load 进来的权重整个冲掉。要改 n_steps/batch_size 只能重建 rollout buffer（见下）。
        changed = {}
        for name, want in (
            ("n_epochs", train_cfg.n_epochs),
            ("gamma", train_cfg.gamma),
            ("gae_lambda", train_cfg.gae_lambda),
            ("ent_coef", train_cfg.ent_coef),
            ("vf_coef", train_cfg.vf_coef),
            ("max_grad_norm", train_cfg.max_grad_norm),
            ("target_kl", train_cfg.target_kl),
        ):
            got = getattr(model, name, None)
            if got != want:
                changed[name] = (got, want)
                setattr(model, name, want)

        # clip_range 从 load() 回来时是 FloatSchedule 对象、不是 float，而 PPO.train() 里写的是
        # `self.clip_range(progress)`——setattr 成裸 float 就会在第一次更新时抛
        # TypeError: 'float' object is not callable。所以必须重新包一层。
        # learning_rate 反过来，load 回来就是裸 float，直接 setattr 即可，但 lr_schedule 要一起换。
        if _lr_or_clip_value(model.learning_rate) != train_cfg.learning_rate:
            changed["learning_rate"] = (_lr_or_clip_value(model.learning_rate), train_cfg.learning_rate)
            model.learning_rate = train_cfg.learning_rate
            model.lr_schedule = FloatSchedule(train_cfg.learning_rate)
        if _lr_or_clip_value(model.clip_range) != train_cfg.clip_range:
            changed["clip_range"] = (_lr_or_clip_value(model.clip_range), train_cfg.clip_range)
        model.clip_range = FloatSchedule(train_cfg.clip_range)
        if train_cfg.clip_range_vf is None:
            model.clip_range_vf = None
        else:
            model.clip_range_vf = FloatSchedule(train_cfg.clip_range_vf)

        if model.n_steps != train_cfg.n_steps:
            changed["n_steps"] = (model.n_steps, train_cfg.n_steps)
            model.n_steps = train_cfg.n_steps
        if model.batch_size != train_cfg.batch_size:
            changed["batch_size"] = (model.batch_size, train_cfg.batch_size)
            model.batch_size = train_cfg.batch_size
        # rollout buffer 不在 checkpoint 里（SB3 的 _excluded_save_params），load 时本来就会重建，
        # 所以这里重建它不丢东西。顺序必须在上面几个超参设完之后：buffer 要拿新的 gamma/gae_lambda。
        model.rollout_buffer = model.rollout_buffer_class(
            model.n_steps, model.observation_space, model.action_space,
            device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
            n_envs=model.n_envs, **model.rollout_buffer_kwargs,
        )
        if changed:
            print("  续训覆盖超参（checkpoint 里的值 -> 本次指定的值）：")
            for k, (g, w) in changed.items():
                print(f"    {k}: {g} -> {w}")
        if model.n_steps * model.n_envs % model.batch_size:
            print(f"  ! n_steps×n_envs = {model.n_steps * model.n_envs} 不是 batch_size "
                  f"{model.batch_size} 的整数倍，每轮最后一个 minibatch 是零头（SB3 会警告）")
        reset_num_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            device="cpu",
            learning_rate=train_cfg.learning_rate,
            n_steps=train_cfg.n_steps,
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            clip_range=train_cfg.clip_range,
            clip_range_vf=train_cfg.clip_range_vf,
            ent_coef=train_cfg.ent_coef,
            vf_coef=train_cfg.vf_coef,
            max_grad_norm=train_cfg.max_grad_norm,
            target_kl=train_cfg.target_kl,
            policy_kwargs=dict(net_arch=list(train_cfg.net_arch), log_std_init=train_cfg.log_std_init),
            tensorboard_log=str(tb_dir),
            seed=train_cfg.seed,
            verbose=0,
        )
        reset_num_timesteps = True

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, train_cfg.checkpoint_freq // args.n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="rl",
            verbose=0,
        ),
        EvalMetricsCallback(
            env_cfg, reward_cfg, train_cfg.eval_freq, train_cfg.eval_episodes,
            save_dir=run_dir / "best", verbose=1,
        ),
    ]

    print(f"开始训练（TensorBoard: tensorboard --logdir {RUNS_DIR}）…")
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False,
                reset_num_timesteps=reset_num_timesteps)
    elapsed = time.perf_counter() - t0

    model.save(str(run_dir / "model.zip"))
    vec_env.close()
    print(f"训练结束：{elapsed / 60:.1f} 分钟，实测 {args.steps / elapsed:,.0f} 步/秒")
    print(f"模型已保存到 {run_dir / 'model.zip'}")
    print(f"回放：python3 -m go2_ppo.play --mode viewer --model {run_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
