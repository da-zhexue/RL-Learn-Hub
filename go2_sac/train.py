"""用 SAC 训练 Go2 过地形。

用法（在仓库根目录，两种写法都行）：
    python3 -m go2_sac.train --terrain full --steps 3_000_000
    python3 go2_sac/train.py --terrain flat --steps 1_000_000 --n-envs 8

建议的课程（先易后难，比重头硬啃台阶快很多）：
    1) --terrain flat  训练到能稳定小跑不摔        （约 1e6 步 / 约 38 分钟）
    2) --terrain steps 微调，学会抬腿过 0.08 m 的槛 （约 1e6 步）
    3) --terrain full  微调爬台阶                  （越往后越慢）
    4) --resume 把上一阶段的 model.zip 接着训

实测吞吐：8 个环境、CPU、无 GPU，稳态约 440 环境步/秒（瓶颈在 SAC 的梯度更新，不在物理：
环境本身能跑 6300 步/秒）。所以 1e6 步约 38 分钟，3e6 步约 1.9 小时。

注意并行环境数比线程数关键：n_envs=2 时只有约 120 步/秒，n_envs=8 时约 440~620 步/秒。
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
from dataclasses import replace  # noqa: E402
from functools import partial  # noqa: E402

# 支持 `python3 go2_sac/train.py` 这种直接运行（此前踩过 "No module named 'ctrl'" 的坑）
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.buffers import ReplayBuffer  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.utils import get_schedule_fn  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from go2_common.config import (  # noqa: E402
    DEFAULT_SCENE,
    MODELS_DIR,
    RUNS_DIR,
    EnvCfg,
    RewardCfg,
    obs_dim,
    to_jsonable,
)
from go2_common.env import Go2TerrainEnv  # noqa: E402
from go2_common.terrain import TERRAINS  # noqa: E402
from go2_common.train_utils import (  # noqa: E402
    CONFIG_NAME,
    EvalMetricsCallback,
    checkpoint_obs_dim,
    load_resume_configs,
    make_env,
    pad_replay_buffer,
    resolve_model_path,
    warm_start_scan,
)
from go2_sac.config import TrainCfg  # noqa: E402


# --------------------------------------------------------------- 演示数据预热
#
# 先用开环小跑采一批转移塞进 replay buffer（SACfD / demonstration seeding），
# **只给 Q 一个起点**，策略之后仍然完全自由地学。为什么必须这么做见 AGENT.md §6.4。

# 执行器顺序 FR, FL, RR, RL；对角小跑 FR+RL 同相、FL+RR 反相
_TROT_PHASE = np.repeat(np.array([0.0, np.pi, np.pi, 0.0]), 3)

# 预热数据**重新标注奖励**：只保留任务项，这些姿态/能耗项一律置零。
# `trot_action` 只驱动大腿和小腿、没有髋关节，必然自转横漂，原样计费会把预热数据
# 变成负分（理由和实测数字见 AGENT.md §5-B6）。
# `lateral_vel` 同理（2026-10-10 加）：手调小跑的 vy²=0.0462，按 −5.0 计是 −0.23/步，
# 一条 1000 步的预热轨迹要倒扣 ~230 分，而任务项（progress+track）整条才 ~190 分——
# 不置零就是把预热数据全变成负样本。
POSTURE_TERMS = (
    "base_height", "lateral", "lateral_vel", "orientation", "yaw", "yaw_rate",
    "action_rate", "torques", "collision",
)


def trot_action(t: int, freq: float, amp: float, phase: float, control_hz: float) -> np.ndarray:
    """手调的对角小跑，只用于预热 replay buffer（实测 ±0.25 rad、3 Hz 时 10 秒走 4.57 m）。"""
    n = max(int(round(control_hz / freq)), 2)
    ang = 2.0 * np.pi * ((t % n) / n) + phase
    s = np.sin(ang + _TROT_PHASE)
    a = np.zeros(12)
    a[1::3] = amp * s[1::3]  # 大腿
    a[2::3] = amp * s[2::3]  # 小腿
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def collect_seed_transitions(env_cfg, reward_cfg, n_trot, n_rand, seed=0):
    """采一批开环小跑（+ 少量站立/随机动作）的转移用于预热。单进程跑，几十秒。

    奖励用 `reward_cfg` 的**任务项**，姿态/能耗项按 POSTURE_TERMS 置零。
    """
    demo_cfg = replace(reward_cfg, **{k: 0.0 for k in POSTURE_TERMS})
    env = Go2TerrainEnv(env_cfg, demo_cfg)
    control_hz = 1.0 / (env_cfg.sim_dt * env_cfg.decimation)
    rng = np.random.default_rng(seed)
    obs_l, next_l, act_l, rew_l, done_l, timeout_l = [], [], [], [], [], []
    for ep in range(n_trot + n_rand):
        obs, _ = env.reset(seed=seed + ep)
        # 每条轨迹随机化频率/幅度/相位，别只喂一条确定性轨迹
        # 幅度按**绝对关节摆幅**（0.24~0.40 rad）给，再除以 action_scale 折算成动作值：
        # 这样换 action_scale 时预热的小跑还是同一个物理步态。直接写动作幅度（曾用 0.6~1.0）
        # 会随 action_scale 一起放大——0.6 时摆幅冲到 0.6 rad，狗直接摔，采集量从 3.5 万条掉到 1.1 万条。
        freq = rng.uniform(2.0, 3.5)
        amp = rng.uniform(0.24, 0.40) / env_cfg.action_scale
        phase = rng.uniform(0, 2 * np.pi)
        for t in range(env.max_steps):
            if ep < n_trot:
                a = trot_action(t, freq, amp, phase, control_hz)
            elif ep % 2:
                a = np.zeros(12, dtype=np.float32)  # 混一点"站着"，让 Q 能对比出高下
            else:
                a = rng.normal(0.0, 0.3, 12).astype(np.float32)
            next_obs, rew, term, trunc, _ = env.step(a)
            obs_l.append(obs)
            next_l.append(next_obs)
            act_l.append(a)
            rew_l.append(rew)
            done_l.append(float(term or trunc))
            timeout_l.append(bool(trunc and not term))
            obs = next_obs
            if term or trunc:
                break
    env.close()
    return (np.asarray(obs_l), np.asarray(next_l), np.asarray(act_l),
            np.asarray(rew_l, dtype=np.float32), np.asarray(done_l, dtype=np.float32),
            np.asarray(timeout_l, dtype=bool))


def seed_replay_buffer(model, vec_env, env_cfg, reward_cfg, n_trot, n_rand, seed=0):
    """把预热数据填进 model 的 replay buffer，返回 (条数, 平均每步回报)。"""
    obs, next_obs, act, rew, done, timeouts = collect_seed_transitions(
        env_cfg, reward_cfg, n_trot, n_rand, seed
    )
    n_envs = vec_env.num_envs
    n = (len(obs) // n_envs) * n_envs  # buffer 按 n_envs 分批存，截到整数倍
    rb = ReplayBuffer(
        model.buffer_size, vec_env.observation_space, vec_env.action_space,
        device=model.device, n_envs=n_envs, handle_timeout_termination=True,
    )
    for i in range(0, n, n_envs):
        sl = slice(i, i + n_envs)
        infos = [{"TimeLimit.truncated": bool(timeout)} for timeout in timeouts[sl]]
        rb.add(obs[sl], next_obs[sl], act[sl], rew[sl], done[sl], infos)
    model.replay_buffer = rb
    return n, float(rew.mean())


def prepare_resume_buffer(model, replay_buffer_path, reset_buffer, learning_starts):
    """Reject replay with lost timeout labels; never infer them from episode length."""
    if replay_buffer_path is not None and not reset_buffer:
        model.load_replay_buffer(str(replay_buffer_path))
        if not model.replay_buffer.handle_timeout_termination:
            raise ValueError(
                f"{replay_buffer_path} 的超时标签已丢失，不能安全续用。"
                "请加 --reset-replay-buffer 保留模型权重、清空旧数据并重新采集；"
                "原 .pkl 文件不会被修改。"
            )
        print(f"  已从 {replay_buffer_path} 加载 replay buffer（超时处理已启用）")
    else:
        model.replay_buffer.reset()
        model.replay_buffer.handle_timeout_termination = True
        model.replay_buffer.timeouts.fill(0)
        print("  replay buffer 从空开始，保留模型权重和优化器，重新采集数据")
    if model.replay_buffer.size() == 0:
        # SB3 compares learning_starts against cumulative num_timesteps on resume.
        model.learning_starts = model.num_timesteps + learning_starts
        print(f"  先采集 {learning_starts:,} 步，再开始梯度更新")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="用 SAC 训练 Go2 过地形（平地 -> 槛 -> 台阶）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 这几个默认 None = "没显式指定"，这样 --resume 时才能让被续训模型的 config.json 说了算
    p.add_argument("--terrain", choices=TERRAINS, default=None,
                   help="课程阶段 flat/steps/full；默认不指定时用 full（--resume 则沿用模型自己的）")
    p.add_argument("--scene", default=None, help="场景 xml，默认 unitree_mujoco 自带的 scene.xml")
    p.add_argument("--steps", type=int, default=TrainCfg.total_steps, help="总训练步数")
    p.add_argument("--n-envs", type=int, default=TrainCfg.n_envs, help="并行环境数（SAC 下主要买的是多样性，不是速度）")
    p.add_argument("--seed", type=int, default=TrainCfg.seed)
    p.add_argument("--save-dir", default=None, help="默认 models/go2_sac_<地形>_<时间戳>/")
    p.add_argument("--resume", default=None,
                   help="从已有的模型接着训，.zip 可省略；会读它同目录的 config.json（也会往上找一层，"
                        "所以 checkpoints/rl_xxx 这种路径直接给就行）")
    p.add_argument("--reset-replay-buffer", action="store_true",
                   help="续训时不加载旧 replay buffer，保留模型权重，先重新采集数据；"
                        "超时标签已丢失的旧 buffer 必须使用此选项")
    p.add_argument("--privileged", action="store_true", default=None,
                   help="观测里加 1 维机身离地高度（真机上没有，仅作对照）")
    p.add_argument("--scan", action="store_true", default=None,
                   help="观测里加 16 维前方地形高度扫描（47 -> 63 维）。真机要靠深度相机+"
                        "高程图提供，本仓库没实现，所以只用于验证'卡点是不是信息'。"
                        "续训 47 维 checkpoint 时自动暖启动：新增列填 0，策略逐位不变，"
                        "旧 replay buffer 因维度不符会被清空")
    p.add_argument("--reset-x", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="起点 x 随机范围，例如 --reset-x 1.0 2.0 直接练爬台阶")
    p.add_argument("--reset-frac-stair", type=float, default=None, metavar="F",
                   help="用近台阶窗口（默认 1.40~1.85）开局的回合比例，0=关闭（默认）。"
                        "实测平地占满一回合 1000 步里的 98.5%%，接触台阶的只有 48 步；"
                        "调大能把爬台阶的梯度从平地步态里拎出来。注意 >0 会换一条 RNG 流，"
                        "评估环境永远钉死为 0，所以选模 key 仍然可比")
    p.add_argument("--reset-x-stair", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="近台阶出生窗口，配合 --reset-frac-stair。1.85+0.24(脚印半长)=2.09 < 2.10，"
                        "整只狗还在第一个立面之前，不会一出生就站在台阶上")
    p.add_argument("--terrain-scale", type=float, default=None, metavar="S",
                   help="地形高度整体缩放（课程用）：0.3 / 0.6 / 1.0 逐级长高，1.0=原场景，0=平地。"
                        "x/y 脚印不变，只压矮高度")
    p.add_argument("--action-scale", type=float, default=None, metavar="A",
                   help="关节目标幅度，决定能跨多高的槛。续训时必须显式给，否则沿用 checkpoint 的值（见 AGENT.md §5-A2）")
    p.add_argument("--learning-rate", type=float, default=None, metavar="LR",
                   help="微调时调小（如 1e-4）。续训时不显式给会被 checkpoint 的值静默覆盖（见 AGENT.md §5-A1）")
    p.add_argument("--target-entropy", type=float, default=None, metavar="H",
                   help="SAC 的目标熵，从零训练默认 -6.0（SB3 默认 -12 会压掉探索，见 AGENT.md §6.4）")
    p.add_argument("--gradient-steps", type=int, default=None, metavar="G",
                   help="每收集一轮（= train_freq × n_envs = 8 个环境步）做几次梯度更新，"
                        "不是「每次更新几步」（见 AGENT.md §5-A4 和 TrainCfg.gradient_steps）")
    p.add_argument("--fresh-reward", action="store_true",
                   help="续训时**不**沿用 checkpoint 里的奖励权重，改用当前 RewardCfg 的默认值；"
                        "换课程阶段必须加（见 AGENT.md §5-A2）")
    p.add_argument("--eval-freq", type=int, default=None, metavar="N",
                   help="每 N 个环境步评估一次（默认 50000）")
    p.add_argument("--eval-episodes", type=int, default=None, metavar="N",
                   help="每次评估跑几个回合。**默认 20，别往小调**（理由见 AGENT.md §5-D2）")
    p.add_argument("--eval-ref-scales", type=float, nargs="+", default=None, metavar="S",
                   help="每次评估额外跑的参考地形尺度（只进日志，不参与选模）。默认 (0.56, 0.70)。"
                        "爬课程梯子时**必须按当前档位改**：和当前档相同的尺度会被自动跳过，"
                        "若不改，升到 0.70 之后参考尺度就只剩 0.56，**没有向上看的窗口了**——"
                        "而 R3 正是靠那个窗口（0.70 从 0/20 涨到 13/20）才确认训练真的在进步。"
                        "建议给两个：下方一档（灾难性遗忘的哨兵）+ 上方一档（前瞻）")
    p.add_argument("--checkpoint-freq", type=int, default=None, metavar="N",
                   help="每 N 个环境步存一个 checkpoint（默认 200000）")
    p.add_argument("--torch-threads", type=int, default=TrainCfg.torch_threads)
    p.add_argument("--seed-trot", type=int, default=None, metavar="N",
                   help="用 N 条开环小跑轨迹预热 replay buffer（原理见 AGENT.md §6.4）；0 = 关闭。"
                        "不写时：从头训练默认开，**续训默认关**——预热喂的手调小跑比已经会走路的"
                        "策略差得多，拿它当先验等于把好策略往回拽")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.reset_replay_buffer and not args.resume:
        raise ValueError("--reset-replay-buffer 只能与 --resume 一起使用")
    torch.set_num_threads(args.torch_threads)

    # 先确定环境：--resume 时以模型的 config.json 为准，命令行显式给的参数再覆盖它。
    # 顺序不能反——反过来会出现"想练台阶但环境还是平地"这种静默错误。
    resume_path, replay_buffer_path = (None, None) if not args.resume else resolve_model_path(args.resume)
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
        if args.scan is not None:
            env_cfg.scan = bool(args.scan)
        if args.reset_x is not None:
            env_cfg.reset_x = tuple(args.reset_x)
        if args.reset_x_stair is not None:
            env_cfg.reset_x_stair = tuple(args.reset_x_stair)
        if args.reset_frac_stair is not None:
            env_cfg.reset_frac_stair = min(max(float(args.reset_frac_stair), 0.0), 1.0)
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
            scan=bool(args.scan),
        )
        if args.reset_x is not None:
            env_cfg.reset_x = tuple(args.reset_x)
        if args.reset_x_stair is not None:
            env_cfg.reset_x_stair = tuple(args.reset_x_stair)
        if args.reset_frac_stair is not None:
            env_cfg.reset_frac_stair = min(max(float(args.reset_frac_stair), 0.0), 1.0)
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
        else MODELS_DIR / f"go2_sac_{env_cfg.terrain}{scale_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # 观测维度变了（这次新开地形扫描，47 -> 63）时，**不能**拿旧 checkpoint 直接 SAC.load：
    # 第一层权重的形状对不上，SB3 会当场报错。先把 checkpoint 扩成新维度再续训，
    # 新增的输入列填 0 —— 这样暖启动那一刻策略和旧模型**逐位相同**（有单测，
    # 见 go2_common/tests/test_training_fixes.py），新维度只能靠梯度自己长进来。
    #
    # 必须做在下面"种 best"**之前**：种进去的要是扩维后的模型，否则评估回调加载
    # 那个 47 维的 inherited_best 时会因为维度不符炸掉。
    force_reset_buffer = False
    if resume_path is not None:
        want_dim = obs_dim(env_cfg.privileged, env_cfg.scan)
        got_dim = checkpoint_obs_dim(resume_path)
        if got_dim != want_dim:
            warm_path = run_dir / "warm_start" / "best_model.zip"
            warm_start_scan(resume_path, warm_path, want_dim)
            print(f"  暖启动：观测 {got_dim} -> {want_dim} 维"
                  f"（新增 {want_dim - got_dim} 维地形扫描，权重填 0，策略逐位不变）")
            print(f"  暖启动模型：{warm_path}")
            resume_path = warm_path
            # 旧 buffer 存的是 got_dim 维观测，维度不符。
            # **补齐，不要清空**：2026-10-13 实测清空两次都塌（R7、R12），
            # R12 在 25k 步内 actor_loss 冲到 −126、成功率 8/20 → 0/20，
            # 而同样条件不清 buffer 的 R11 完全正常。机理见 pad_replay_buffer 的注释。
            if replay_buffer_path is not None and not args.reset_replay_buffer:
                padded = run_dir / "warm_start" / "replay_buffer_padded.pkl"
                pad_replay_buffer(replay_buffer_path, padded, want_dim)
                print(f"  replay buffer 补齐到 {want_dim} 维（新增列填 0，"
                      f"保留 {padded.stat().st_size / 1e6:.0f} MB 旧经验）：{padded}")
                replay_buffer_path = padded
            else:
                replay_buffer_path = None
                force_reset_buffer = True

    # 换档续训（--save-dir 指向新目录）时，先把续训起点种进新目录的 best/。
    # 不种的话，新目录里 `_bootstrap_best` 的两个分支都会落空（既没有 best_metric.json，
    # 也没有 best/best_model.zip），于是**第一次评估**——它发生在 25k 步之后、权重已经被
    # 练过一轮——会直接当上基准。万一那 25k 步把策略练差了，新档的 best_model 就会比
    # 继承来的那个还差，而这正是 §2.1 那条 key 要防的事。种进去之后 `_bootstrap_best`
    # 会按**当前口径**把它重新量一遍当基准（口径不同，不会误用旧 key），换档不再丢东西。
    inherited_best = run_dir / "best" / "best_model.zip"
    if resume_path is not None and not inherited_best.exists():
        inherited_best.parent.mkdir(parents=True, exist_ok=True)
        inherited_best.write_bytes(pathlib.Path(resume_path).read_bytes())
        print(f"  已把续训起点种进 {inherited_best}（换档时用来立选模基准）")

    train_cfg = TrainCfg(total_steps=args.steps, n_envs=args.n_envs, seed=args.seed)
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.target_entropy is not None:
        train_cfg.target_entropy = args.target_entropy
    if args.gradient_steps is not None:
        train_cfg.gradient_steps = args.gradient_steps
    if args.eval_freq is not None:
        train_cfg.eval_freq = args.eval_freq
    if args.eval_episodes is not None:
        train_cfg.eval_episodes = args.eval_episodes
    if args.eval_ref_scales is not None:
        train_cfg.eval_reference_scales = tuple(args.eval_ref_scales)
    if args.checkpoint_freq is not None:
        train_cfg.checkpoint_freq = args.checkpoint_freq
    if args.seed_trot is not None:
        train_cfg.seed_trot_episodes = args.seed_trot
    elif resume_path is not None:
        # 续训默认不预热：buffer 里已经有上一阶段的真实数据，再灌开环小跑是往回拽
        train_cfg.seed_trot_episodes = 0

    # 把配置落盘：play.py 靠它还原环境，避免观测维度和奖励权重对不上
    with open(run_dir / CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(
            {"env": to_jsonable(env_cfg), "reward": to_jsonable(reward_cfg), "train": to_jsonable(train_cfg)},
            f, indent=2, ensure_ascii=False,
        )
    print(f"地形={env_cfg.terrain}(scale={env_cfg.terrain_scale:g})  action_scale={env_cfg.action_scale:g}  "
          f"观测维度={obs_dim(env_cfg.privileged, env_cfg.scan)}"
          f"{'(含 16 维地形扫描)' if env_cfg.scan else ''}  "
          f"环境数={args.n_envs}  总步数={args.steps:,}  输出目录={run_dir}")

    # 每个 --save-dir 一个 tb 子目录（全写进 runs/ 会把不同 run 的曲线混成一条，见 AGENT.md §5-A5）
    tb_dir = RUNS_DIR / run_dir.name
    tb_dir.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv(
        [partial(make_env, env_cfg, reward_cfg, args.seed + i) for i in range(args.n_envs)]
    )

    # 有预热数据时立刻开始梯度更新（buffer 里已经有东西了），否则按 learning_starts 等真实数据。
    # **续训时把预热关掉**（--seed-trot 0）：预热喂的是手调开环小跑，它比一个已经会走路的
    # 策略差得多（后腿抬不起来、没有航向控制），拿它当先验等于把好策略往回拽。
    learning_starts = 0 if train_cfg.seed_trot_episodes > 0 else train_cfg.learning_starts

    if resume_path is not None:
        model = SAC.load(str(resume_path), env=vec_env, device="cpu", tensorboard_log=str(tb_dir))
        try:
            prepare_resume_buffer(
                model,
                replay_buffer_path,
                args.reset_replay_buffer or force_reset_buffer,
                train_cfg.learning_starts,
            )
        except Exception:
            vec_env.close()
            raise
        # 续训必须显式把超参再盖回去：SAC.load() 会还原 checkpoint 里的超参。
        #
        # 但**不能拿 train_cfg 直接比**：TrainCfg 的数据类默认值不是"用户本次指定的值"，
        # 没传 CLI 开关的项会被静默改回默认值。实测踩过：续训时不写 --learning-rate，
        # checkpoint 里的 1e-4 被无声换成默认 3e-4，两次评估内就把爬台阶的能力丢掉。
        # 所以凡是**有 CLI 开关**的项，一律只在显式传了的时候才覆盖；没有 CLI 开关的项
        # （batch_size/tau/gamma/train_freq）本来也只能靠改 config.py 来调，仍以 config 为准。
        changed = {}
        for name, want, explicit in (
            ("learning_rate", train_cfg.learning_rate, args.learning_rate is not None),
            ("batch_size", train_cfg.batch_size, True),
            ("tau", train_cfg.tau, True),
            ("gamma", train_cfg.gamma, True),
            ("gradient_steps", train_cfg.gradient_steps, args.gradient_steps is not None),
            ("target_entropy", train_cfg.target_entropy, args.target_entropy is not None),
            ("train_freq", train_cfg.train_freq, True),
        ):
            if not explicit:
                continue
            got = getattr(model, name, None)
            if got != want:
                changed[name] = (got, want)
                setattr(model, name, want)
        if "learning_rate" in changed:
            model.lr_schedule = get_schedule_fn(train_cfg.learning_rate)
        # train_freq 必须是 TrainFreq 对象，直接塞 int 会在采样时炸
        model._convert_train_freq()
        if changed:
            print("  续训覆盖超参（checkpoint 里的值 -> 本次指定的值）：")
            for k, (g, w) in changed.items():
                print(f"    {k}: {g} -> {w}")
        # 无论覆盖与否都打印**实际生效**的值：上面那个坑就是靠这行才发现的
        print(f"  续训生效超参：lr={model.learning_rate:g} (lr_schedule(1)={model.lr_schedule(1):g})  "
              f"gamma={model.gamma:g}  tau={model.tau:g}  batch={model.batch_size}  "
              f"gradient_steps={model.gradient_steps}  target_entropy={model.target_entropy}  "
              f"train_freq={model.train_freq.frequency}")
        # buffer_size 不覆盖：buffer 已经按旧容量分配好了，中途改会丢数据，收益也不大
        reset_num_timesteps = False
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            device="cpu",
            learning_rate=train_cfg.learning_rate,
            buffer_size=train_cfg.buffer_size,
            learning_starts=learning_starts,
            batch_size=train_cfg.batch_size,
            tau=train_cfg.tau,
            gamma=train_cfg.gamma,
            train_freq=train_cfg.train_freq,
            gradient_steps=train_cfg.gradient_steps,
            ent_coef="auto",
            target_entropy=train_cfg.target_entropy,
            policy_kwargs=dict(net_arch=list(train_cfg.net_arch), log_std_init=train_cfg.log_std_init),
            tensorboard_log=str(tb_dir),
            seed=train_cfg.seed,
            verbose=0,
        )
        reset_num_timesteps = True

    if train_cfg.seed_trot_episodes > 0:
        t_seed = time.perf_counter()
        n_seed, seed_rew = seed_replay_buffer(
            model, vec_env, env_cfg, reward_cfg,
            train_cfg.seed_trot_episodes, train_cfg.seed_rand_episodes, seed=train_cfg.seed,
        )
        print(f"预热 replay buffer：{n_seed:,} 条转移，平均 {seed_rew:+.3f} 分/步"
              f"（{time.perf_counter() - t_seed:.0f} s）")

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, train_cfg.checkpoint_freq // args.n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="rl",
            save_replay_buffer=True,
            verbose=0,
        ),
        EvalMetricsCallback(
            env_cfg, reward_cfg, train_cfg.eval_freq, train_cfg.eval_episodes,
            save_dir=run_dir / "best", verbose=1, save_buffer=True,
            reference_scales=train_cfg.eval_reference_scales,
            reference_episodes=train_cfg.eval_reference_episodes,
        ),
    ]

    print(f"开始训练（TensorBoard: tensorboard --logdir {RUNS_DIR}）…")
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False,
                reset_num_timesteps=reset_num_timesteps)
    elapsed = time.perf_counter() - t0

    model.save(str(run_dir / "model.zip"))
    model.save_replay_buffer(str(run_dir / "model.pkl"))
    vec_env.close()
    print(f"训练结束：{elapsed / 60:.1f} 分钟，实测 {args.steps / elapsed:,.0f} 步/秒")
    print(f"模型已保存到 {run_dir / 'model.zip'}（replay buffer：同目录 model.pkl）")
    print(f"回放：python3 -m go2_sac.play --mode viewer --model {run_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
