# RUNBOOK：在有显卡的机器上把 go2_issac 跑起来

**这份文档是给"那台有 NVIDIA 显卡的机器"看的。当前这台开发机跑不了，一行都跑不了。**

实测这台机器为什么不行（2026-10-07）：

| 项 | 当前机器 | Isaac Sim 要求 |
|---|---|---|
| 显卡 | `nvidia-smi` 不存在，只有 AMD HawkPoint1 核显 | NVIDIA RTX（CUDA） |
| 内存 | 27 GB | ≥ 32 GB |
| 磁盘 `/` 可用 | 21 GB | ~50 GB（Sim + Lab + 缓存） |
| Python | 3.10.12 | 3.11 |

**Isaac Sim 没有任何 CPU 回退**。`--cpu` 那个开关是"用 CPU 跑张量"，物理和渲染照样要 GPU。
所以别在这台机器上试装，会白花几个小时。

---

## 0. 上机前先想清楚：能一致的和不能一致的

| | 能不能和 MuJoCo 侧一致 | 靠什么保证 |
|---|---|---|
| 地形几何、脚下高度、第几级台阶 | **逐位一致** | `course.py` + `tests/course_parity.py`（本机已对拍通过） |
| 45 维观测、14 项奖励、终止/成功判据 | **逐位一致**（1e-15 级） | `core.py` + `tests/core_parity.py`（本机已对拍通过） |
| 资产质量 / 惯量 / 关节限位 / 力矩上限 | 一致 | `assets/go2_reference.json`（从编译后的 MjModel 导出）+ `smoke.py --check-asset` |
| 接触动力学 | **只能做到统计上接近** | 没法保证，见下 |
| 训练曲线、模型权重 | **不可比、不可互用** | —— |

接触那一栏展开说：MuJoCo 用 `elliptic` 摩擦锥 + `impratio=100`，脚底球 `condim=6`
（带扭转和滚动摩擦）、`priority=1` 独占接触对；PhysX 是各向同性的库仑摩擦，没有
priority 这回事。**这是两个不同的物理，不是同一个物理的两种实现。**

由此推出一条硬规矩：**跨仿真器不能接着训**。

> `go2_ppo` 的 `.zip` 和 Isaac 的 `.pt` 互不通用，`--resume` 会直接拒绝 `.zip`。
> 把一个在 MuJoCo 里练到 90% 成功率的策略丢进 Isaac，不会"接着变好"，
> 只会因为物理差异立刻表现崩坏，然后在错误的方向上被继续训练。
> 曲线也只能比趋势（成功率、最高台阶、vx/指令比），不能比数值。

---

## 1. 装环境

**两条路，选一条。** 装完第一件事：`git tag`/记下版本号，然后写进 manifest（第 2 步自动做）。

### 路 A：conda（推荐，出问题好查）

```bash
conda create -n isaac python=3.11 -y && conda activate isaac

# torch 要装 CUDA 版；cu128 对应驱动 >= 525（Sim 5.1 要求驱动 >= 580.65.06，实测 570 也能跑）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Isaac Sim
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

# Isaac Lab：**固定一个 tag**，别用 main（API 在 2.3 和 3.x 之间变过，见下面的差异表）
git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh -i rsl_rl
```

`./isaaclab.sh -i rsl_rl` 会把 `rsl_rl` 和 `isaaclab_rl` 一起装好（`train.py` 用的
`RslRlVecEnvWrapper` / `OnPolicyRunner` 都在这两个包里）。

### 路 B：容器（省事，但要 `--gpus all`）

```bash
docker run --name isaac-lab --gpus all -it --rm \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /path/to/unitree:/workspace/unitree \
  nvcr.io/nvidia/isaac-lab:2.3.0
```

### 版本差异表（**装哪套就以哪套为准**）

| | Isaac Lab 2.3 / Isaac Sim 5.1 | Isaac Lab 3.x / Isaac Sim 6.x |
|---|---|---|
| MJCF 转换 CLI | `scripts/tools/convert_mjcf.py --fix-base --import-sites` | 同名脚本，参数换成 `--merge-mesh --collision-from-visuals --collision-type` |
| 转换 API | `MjcfConverter` / `MjcfConverterCfg`（底层 `MJCFImporter`） | 同左，底层换成 `mujoco-usd-converter` 的 `MJCFImporter(config).import_mjcf()` |
| 已移除的字段 | `fix_base` / `import_sites` | `self_collision` → `allow_self_collision` |

`convert_assets.py` **只传四个两代都有的字段**（`asset_path` / `usd_dir` /
`usd_file_name` / `force_usd_conversion`），就是为了躲开这张表。

### 把整个仓库搬过去

`go2_issac` 要 import `go2_common`（配置唯一来源）和 `go2_ppo.config`（超参来源），
所以**整个 `unitree/` 目录一起搬**，别只拷 `go2_issac/`。

那台机器上**不需要装** mujoco / stable-baselines3 —— 训练回路一行都不碰它们。
（`assets/*.py` 要 mujoco，但那两个脚本是在当前这台机器上跑的，产物已经进 git 了。）

---

## 2. 转 USD

```bash
python3 go2_issac/convert_assets.py            # 转 go2.xml -> assets/go2.usd
python3 go2_issac/convert_assets.py --scene    # 可选：把 scene.xml 也转一份，纯目视对比
```

跑完会写 `assets/manifest.json`（Isaac Lab / Isaac Sim 版本、USD 的 sha256、MJCF 的 sha256、
仓库 commit）。**这份文件不进 git**，它是"当时那台机器是什么状态"的记录，出问题先看它。

三个容易踩的点，`convert_assets.py` 里已经处理/注释了：

1. **驱动增益不在这步烤进 USD。** MJCF importer 没有 `default_drive_stiffness`
   （那是 URDF 的），`go2.xml` 里的 `<motor>` 只变成 USD 的 DriveAPI。
   kp/kd 由 `env_cfg.py` 的 `ImplicitActuatorCfg` 在实例化时写回：
   `stiffness=60`、`damping=3.6`。**是 3.6 不是 3.5**——MuJoCo 的 12 个铰链自己带
   `damping=0.1`，Isaac 的隐式执行器只有 `damping` 一个旋钮，两个要加起来。
2. **不要传 `fix_base`。** `go2.xml` 有 freejoint，保持浮动基座。
3. **视觉网格不能变成碰撞体。** `go2.xml` 里 33 个 mesh 是 `contype=0 conaffinity=0`
   的纯视觉几何；万一 importer 给它们加了 CollisionAPI，狗会被自己的网格卡住。
   脚本里的 `strip_visual_collision()` 会去掉，`smoke.py --check-asset` 会验个数。

---

## 3. 自检（**这是最重要的一步，别跳**）

```bash
python3 go2_issac/smoke.py --check-env                      # 先跑这个：版本 / 显卡 / 依赖
python3 go2_issac/smoke.py --all --num-envs 4               # 其余五节
```

`--all` 分五节，每节都能单跑：

| 节 | 验什么 | 报错了改哪儿 |
|---|---|---|
| `--check-asset` | USD 实例化出来的质量 / 关节限位（**前腿后腿不一样**）/ 力矩上限 / kp / damping | `env_cfg.py` 的 `CourseSceneCfg`；质量不对就是转换的问题 |
| `--check-order` | 关节顺序、动作→关节目标的换算、`prev_action` 的推进时机 | `mdp/state.py` 的 `JOINT_NAMES` / `raw_action` |
| `--check-terrain` | 环境原点 → 课程坐标的映射（高程图对不对得上） | **只改 `course.py:COURSE_ORIGIN_FROM_ENV_ORIGIN` 这一个常量** |
| `--sanity` | 观测维度、投影重力符号、奖励与 `core.compute_reward` 逐元素一致、超时走 truncated | `env_cfg.py` 里挂的 term |
| `--drop-test` | 零动作静置 3 s，和 MuJoCo 真值比离地高度与脚底力 | kp/kd/质量/地形高度 |

**四件事一确认，剩下的就只是调参了：关节顺序、坐标偏移、落地高度、驱动增益。**

### `--check-terrain` 对不上怎么办

这是唯一一个"只能上机校准"的常量。Isaac Lab 把每个 patch 的**中心**世界坐标写进
`env.scene.env_origins`，但"中心"这个说法在版本之间变过（有过一段是以角点为准）。
`course.py` 里写的是中心口径的偏移量。如果 `--check-terrain` 报出来的课程坐标
不是 `(1.5, 0.0)`，把差值直接加到/减到那个常量上就行，别处不用动。

### 想看高程图长什么样

`--check-terrain` 会打印整张高程图的像素尺寸。想在 viewport 里目视对比，
用 `convert_assets.py --scene` 转出来的 `scene.usd` 单独开一个 stage 看，
和 MuJoCo 的 `scene.xml` 摆一起对。

---

## 4. 训练

```bash
# 1) 平地跑稳
python3 go2_issac/train.py --terrain flat  --steps 200_000 --num-envs 512 --headless

# 2) 过两个 0.08 m 的槛
python3 go2_issac/train.py --terrain steps --steps 100_000 --num-envs 512 --headless

# 3) 爬六级台阶
python3 go2_issac/train.py --terrain full  --steps 150_000 --num-envs 2048 --headless
```

课程顺序和 MuJoCo 侧完全一样，先易后难。`--steps` 是**总转移数**（各环境求和），
和 `go2_ppo` 同口径。

输出 `models/go2_issac_<地形>[_s<缩放>]_<时间戳>/`：
`config.json`（环境 + 奖励 + 超参）、`model_<迭代数>.pt`、TensorBoard 事件。

```bash
tensorboard --logdir models/
```

### 几个和 MuJoCo 侧不一样的旋钮

- **`--num-envs` 是最大的性能旋钮。** GPU 上环境数越多越划算（512 起步，几 GB 显存能上 4096）。
  但**地形显存随环境数平方增长**（行和列都变长），见 `env_cfg._patch_grid` 里的算术：
  512 个环境 = 64 MB 高程图，4096 个 = 512 MB。改之前先算这一笔。
- **`--mini-batches`** 默认按 SB3 口径换算（每个 minibatch 恰好 256 条转移）。
  环境一多这个数会很大，如果发现迭代时间全花在优化器上，往下调。
- **PPO 超参没有 CLI。** 全部集中在 `agents/rsl_rl_ppo_cfg.py`，那里有一张和
  `go2_ppo/config.py` 的逐项映射表。要改就改那张表，别在别处塞第二份默认值。
- **`eval/` 那几个数是训练用的随机策略的统计**，不是确定性策略、也不是独立评估环境。
  Isaac 里再起一个物理场景太贵，改成在训练环境里顺带统计（`mdp/metrics.py`）。
  口径和 MuJoCo 侧的 `EvalMetricsCallback` 逐项对齐，**趋势可比，绝对值不要跨实现比**。

### 续训

```bash
python3 go2_issac/train.py --steps 50_000 --resume models/go2_issac_full_20261007_120000 --headless
```

给目录会自动挑迭代数最大的 `model_<n>.pt`。给 `.zip` 会被直接拒绝（跨仿真器，见 §0）。

---

## 5. 回放

```bash
# 有显示器：直接看
python3 go2_issac/play.py --resume models/go2_issac_full_20261007_120000 --episodes 5

# 没显示器
python3 go2_issac/play.py --resume <目录> --episodes 10 --headless
```

环境参数（哪档地形、缩放、`action_scale`、观测维度）**全部从存档同目录的 `config.json`
还原**，不重新猜。每个回合打印一行：回报 / 最远 x / 速度 / 最高台阶 / 是否成功。

---

## 6. 出事了对怎么办

| 症状 | 大概率原因 |
|---|---|
| 狗一落地就抖/塌 | `damping` 写成了 3.5（应该是 **3.6**）；或 `--check-asset` 报 kp 不对 |
| 狗被自己卡住、原地抽搐 | 视觉网格被当成了碰撞体（`--check-asset` 会报碰撞几何个数） |
| 站在平地也判定"摔倒" | 地形高度查询对不上（`--check-terrain`），导致离地高度算成负的 |
| 训不动、回报不涨 | 先看 `--check-order`：关节顺序错了不会报错，只会学不出来 |
| 环境一多就随机崩/丢接触 | `env_cfg.py` 里 PhysX 那几个 capacity 给够 |
| 曲线和 MuJoCo 侧差很多 | **正常**，见 §0。比趋势，别比数值 |
| 换了 Isaac Lab 版本后 USD 加载失败 | 版本差异表，重转 USD；manifest 里有原来那套版本号 |

## 7. 这次**没做**的事

不写自动地形课程（`curriculum=True` 那条）、不写域随机化、不写 sim2real、
不做 SB3↔rsl_rl 的权重互转（跨仿真器没有意义）。

要部署到真机时，用 rsl_rl 官方的 `runner.export_policy_to_jit()` 导 TorchScript，
再按 `ctrl/` 那套 DDS 桥接过去 —— 那是另一件事，不在这次范围里。
