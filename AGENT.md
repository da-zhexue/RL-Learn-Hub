# AGENT.md

> **这个文件和 README.md 的分工**：README 给人看，只写结论和怎么跑，简略。
> 本文件写**为什么**、**踩过什么雷**、**哪些数字是量出来的**、**什么不能碰**。
> 改本仓库前先读完本文件；写代码时以本文件为准。
>
> 维护约定：**踩到新雷就往对应分类里加一条**（现象 / 根因 / 修法 / 实测数字）。
> 加条目比加代码重要——这个仓库的绝大部分损失来自"静默失效"，不是来自写不出的算法。

---

## 1. 仓库结构

```
/home/lin/unitree/
├── go2_common/           ← 算法无关的共享层，SAC / PPO 共用同一份（**只有这一份**）
│   ├── config.py         常量、EnvCfg/RewardCfg、quat_to_rpy/projected_gravity（不 import mujoco）
│   ├── terrain.py        课程变体构造(build_model)、地形高度查询(TerrainHeight)、判据
│   ├── env.py            Go2TerrainEnv（Gymnasium）+ run_episode
│   ├── reward.py         奖励：每项一个函数 + compute_reward() -> (total, parts)
│   └── train_utils.py    make_env / EvalMetricsCallback / load_resume_configs / resolve_model_path
├── go2_sac/              ← 只有 SAC 专属的东西
│   ├── config.py         SAC 的 TrainCfg
│   ├── train.py          SAC 训练入口（含 diag 用的 trot_action、演示数据预热）
│   ├── play.py           回放/部署（--mode viewer / dds）
│   └── diag.py           诊断子命令：terrain/stance/lift/frontier/policy/why/reward
├── go2_ppo/              ← 同上，算法换成 PPO；无预热，其余与 go2_sac 一一对应
│   ├── config.py         PPO 的 TrainCfg
│   ├── train.py          PPO 训练入口
│   ├── play.py           回放/部署
│   └── diag.py           薄壳：把 go2_sac.diag.load_policy 换成 PPO.load 后转发（实现只有一份）
├── go2_issac/            ← 同一任务迁到 Isaac Sim（算法换 rsl_rl，资产自己转 USD）
│   ├── course.py         地形几何唯一真源，纯 numpy（**不 import mujoco / isaaclab**）
│   ├── core.py           观测 / 14 项奖励 / 终止判据，纯 torch（同上）
│   ├── tests/            本机可跑的逐位对拍：course_parity / core_parity
│   ├── assets/           从编译后的 MjModel 导出的资产真值与静置真值（本机跑，产物进 git）
│   ├── convert_assets.py MJCF -> USD（**只能上机**）
│   ├── env_cfg.py mdp/ agents/ train.py play.py smoke.py   （**只能上机**）
│   └── RUNBOOK.md        上机手册：装机、转换、自检、训练
├── ctrl/                 ← 本仓库自己的代码
│   ├── go2_ctrl.py       Go2Ctrl：DDS 收发 + 500 Hz 独立线程 + 插值平滑 + reset
│   └── ctrl_test.py      DDS 手动测试
├── run_curriculum.sh     地形课程（平地 → 两个槛），A=0.6，逐级放开高度
├── run_stairs.sh         楼梯阶段，A=0.7，0.56→1.00 七级
├── README.md             给人看的简略说明
├── AGENT.md              本文件
├── models/               训练输出（**已 gitignore**，几十 GB）
├── runs/                 TensorBoard（**已 gitignore**）
├── logs/                 nohup 日志（**已 gitignore**）
├── unitree_mujoco/       上游克隆，**不要改**
├── unitree_sdk2_python/  上游克隆，**不要改**
└── unitree_ros2/         上游克隆，**不要改**
```

**上游三个目录一律不改**。要改仿真行为只能通过：
`go2_common/terrain.py` 的 `MjSpec` 运行时改写（删/缩放几何体）、`go2_common/env.py` 的步进逻辑、
`go2_common/config.py` 的参数。奖励/环境**只改 `go2_common/` 里那一份**，两个算法包都跟着变——
不要往 `go2_sac/` 或 `go2_ppo/` 里再抄一份环境或奖励。
**用户的 `scene.xml` 不能改**——地形课程是运行时用 `MjSpec` 编译出来的变体，不落盘。

---

## 2. 运行方式

环境：conda env **`sim`**（Python 3.10.12）。CPU only，16 核。无 GPU。

两种调用方式都支持，`train.py` / `play.py` 顶部有 `if __package__ in (None, ""): sys.path.insert(...)` 兜底：

```bash
python3 -m go2_sac.train ...     # 仓根，推荐
python3 go2_sac/train.py ...     # 也可以
```

常用命令：

```bash
# 从头训练（会自动用开环小跑预热 replay buffer）
python3 -m go2_sac.train --terrain full --terrain-scale 0.35 --action-scale 1.0 \
    --steps 2000000 --gradient-steps 1 --eval-freq 50000 --eval-episodes 20 \
    --checkpoint-freq 100000 --save-dir models/amp1_s035

# 课程 / 楼梯
./run_curriculum.sh models/go2_sac_flat_xxx/checkpoints/rl_800000_steps.zip
./run_stairs.sh models/st_s056_full/best/best_model.zip

# 回放
python3 -m go2_sac.play --mode viewer --model models/xxx/best/best_model.zip
python3 -m go2_sac.play --mode dds --model models/xxx/model.zip   # 另开 unitree_mujoco.py
```

```bash
# PPO：同样的课程、同样的参数名，只是换个模块
python3 -m go2_ppo.train --terrain flat --steps 200000 --checkpoint-freq 40000
python3 -m go2_ppo.play  --mode viewer --model models/go2_ppo_flat_xxx/model.zip
```

```bash
# Isaac Sim：同一套任务，算法换 rsl_rl、资产自己转。**这几条只能在有 NVIDIA 显卡的机器上跑**
python3 -m go2_issac.tests.course_parity     # ← 这两条本机就能跑（对拍）
python3 -m go2_issac.tests.core_parity
python3 go2_issac/convert_assets.py
python3 go2_issac/smoke.py --all --num-envs 4
python3 go2_issac/train.py --terrain flat --steps 200_000 --num-envs 512 --headless
python3 go2_issac/play.py --resume models/go2_issac_flat_xxx
```

`go2_issac/train.py` 的 CLI 比上面两个**少得多**：`--terrain --terrain-scale --action-scale
--steps --num-envs --seed --save-dir --resume --privileged --mini-batches --max-iterations`。
PPO 超参一律不进 CLI，全在 `agents/rsl_rl_ppo_cfg.py`——那里只有一份默认值，
不会和 `go2_ppo/config.py` 漂移。上机步骤见 `go2_issac/RUNBOOK.md`。

`go2_sac.train` 的 CLI：`--terrain --scene --steps --n-envs --seed --save-dir --resume --privileged
--reset-x LO HI --terrain-scale --action-scale --learning-rate --target-entropy --gradient-steps
--fresh-reward --eval-freq --eval-episodes --checkpoint-freq --torch-threads --seed-trot`。

`go2_ppo.train` 的 CLI = 上面**去掉** `--target-entropy --gradient-steps --seed-trot`
（PPO 没有目标熵、没有 UTD 这个旋钮、没有 replay buffer 可预热），**加上**
`--n-steps --batch-size --n-epochs --ent-coef --target-kl`。其余参数的语义逐字相同，
包括"`--steps` 续训是本阶段再练多少步""换阶段必须 `--fresh-reward`"。

**吞吐基线（SAC）**（8 环境、CPU）：
- `gradient_steps=1` → **约 440 环境步/秒**（一次更新约 18 ms，8 步一更新）
- `gradient_steps=2` → 约 220 步/秒；`-1`（UTD=1）→ **约 55 步/秒**
- 据此估时间：2M 步 @ `gradient_steps=1` ≈ 75 分钟。

**PPO 的吞吐还没正经测**（只跑过 512/1024 步的验证跑，秒级、被启动开销主导），
估时间以运行时打印的 `time/fps` 为准，不要拿上面 SAC 的数字套。

---

## 3. 代码规范（改代码必须遵守）

1. **`config.py` 不 import mujoco**。它只依赖 numpy——`play.py` 要读 `config.json`、
   `reward.py` 要能脱离仿真单独自检，都不该被迫建模型。
2. **奖励项名 = `reward.py` 里的函数名 = `RewardCfg` 里的字段名**，一一对应，靠
   `TERMS = (("progress", _progress), ...)` 和 `getattr(cfg, name)` 绑定。
   **加/改奖励只动 `RewardCfg` 的权重，不改函数体**；要加新项就在 `reward.py` 里加函数
   并在 `TERMS` 里登记，同时在 `RewardCfg` 加同名字段。
3. **加新奖励项前先过"能不能被刷分"这一关**。仓库里被否掉的例子：
   - `feet_air_time`：不加速度指令闸门时是"原地踏步"和"冲下悬崖"的经典刷分点；
   - 用 `Δbase_z` 当爬升：抬屁股/弹跳就能刷，所以用 `Δterrain_h`；
   - `exp(-err²)` 形式的 `base_height`：站对了发满额奖励 = 发保底工资，零动作站着就能白拿。
4. **注释写"为什么"，并且带上实测数字**。这个仓库的注释密度是刻意高的——每条结论都要能追溯到
   一次测量。不要写"这里做了 X"这种复述代码的注释。
5. **失败了也要把失败写进注释**。被证伪的假设（比如"critic 塌了导致策略崩"）写清楚是怎么被证伪的，
   否则下一个人（或下一轮的你）会重新踩一遍。
6. **不新增第三方依赖**。可用的只有：mujoco 3.10.0、stable-baselines3 2.9.0、gymnasium、
   torch、numpy 1.26.4。训练必须在 CPU 上跑得动。
7. **不要为了"让 demo 好看"放宽成功判据**。成功 = 站上 0.92 m 顶平台；汇报进度用
   "最高到过第几级台阶"这个副指标，不降标准。

---

## 4. 硬不变量（改了就崩，且往往不报错）

| # | 不变量 | 违反的后果 |
|---|---|---|
| 1 | **关节顺序**：`qpos` 按身体树是 FL,FR,RL,RR；执行器 / DDS `LowCmd`/`LowState` / `go2.xml` 的 sensor 是 FR,FL,RR,RL | PD 反馈接到错误关节，狗一上电就瘫倒 |
| 2 | 必须用 `m.actuator_trnid[:,0]` + `m.jnt_qposadr` / `m.jnt_dofadr` 建地址表（`env.qadr` / `env.vadr`） | 硬编码 `qpos[7:19]` 必错。实测映射 = `[10,11,12, 7,8,9, 16,17,18, 13,14,15]` |
| 3 | 观测 45 维 = 角速度3×0.25 + 投影重力3 + 指令3×2 + (q−q_default)12 + dq12×0.05 + 上一步动作12，最后 clip ±10 | 维度和顺序变了，旧模型全部失效（`config.json` 会存布局，但 `play.py` 不校验） |
| 4 | **观测的每一维都必须能从 DDS `LowState` 复现**。代码里不读 `rpy`、`base 高度`、`foot_force` | `--mode dds` 和 viewer 跑的不是同一个策略，真机部署直接失效。DDS 桥里 `rpy`/`foot_force` 恒为 0 |
| 5 | 时序：`SIM_DT=0.005`（= `simulate_python/config.py` 的 `SIMULATE_DT`）、`DECIMATION=4` → 策略 50 Hz | 改成别的值会让 sim-to-sim 差距变大；`play.py --mode dds` 的 50 Hz 组帧也要跟着改 |
| 6 | PD：每物理子步 `tau = kp*(q_target − q) − kd*dq`，逐执行器 clip 到 `ctrlrange`，kp=60 / kd=3.5 | **kd ≥ 4.5 会激发高频振荡**（实测 `|dq|` 冲到 40 rad/s，狗原地抖动并后退） |
| 7 | 默认站姿 `DEFAULT_Q = [0, 0.9, −1.8] * 4`（FR,FL,RR,RL 顺序），取自 `go2.xml` 的 `home` 关键帧 | 顺序错了就是第 1 条的后果 |
| 8 | 超时用 `truncated` 而不是 `terminated` | SB3 才会正确做 bootstrap；用 `terminated` 会让 Q 在超时处被截断 |

---

## 5. 踩过的雷

### A. 静默失效类（最危险：不报错、看起来生效、实际没生效）

**A1 —— 续训时命令行给的超参全部被静默忽略。**
- 现象：同一个模型分别用 `--learning-rate 2e-4` 和 `1e-4` 续训 5 万步，**两次评估数字逐位相同**
  （177.2 / 62.8 / 138.4）。
- 根因：`SAC.load()` 走 `cls(...) → model.__dict__.update(data) → model.__dict__.update(kwargs)`。
  checkpoint 里存的超参在 `update(data)` 时整个恢复，而 `SAC.load()` **不传 kwargs**，
  所以构造函数参数没机会覆盖。
- 修法（`train.py` 续训段）：显式比对 + `setattr`，覆盖
  `learning_rate / batch_size / tau / gamma / gradient_steps / target_entropy / train_freq`。
  **光赋值 `learning_rate` 还不够**——SB3 里它**不是 property**，每个梯度步真正生效的是
  `SAC.train()` 里的 `_update_learning_rate([actor.optimizer, critic.optimizer, ent_coef_optimizer])`，
  它读的是 `self.lr_schedule(...)`。所以必须连 `model.lr_schedule = get_schedule_fn(lr)` 一起换，
  否则三个优化器一个都不动。
- 影响：课程脚本每一级都写着 `LR=2e-4`，实际一直跑的是建模型时的 3e-4；
  而 3e-4 恰好是微调最差的那个（实测 62.8 分 / 1 级台阶，对比 1e-4 的 122.5 分 / 6 级）。
  **这就是"训练反而把模型练坏"的主因之一。**

**A2 —— 续训时 checkpoint 里的旧奖励权重会覆盖 `config.py` 的改动。**
- 现象：改了 `climb` 默认值、给 `base_height` 加了死区，续训后策略行为毫无变化。
- 根因：同 A1，`load_resume_configs` 从 checkpoint 的 `config.json` 恢复奖励权重，
  **只认字段名不认代码**。
- 修法：**换课程阶段必须加 `--fresh-reward`**（用当前 `RewardCfg` 默认值，不沿用 checkpoint 的）。
- 同类：**换阶段必须显式写 `--terrain`**，否则沿用被续训模型自己的。

**A3 —— `--steps` 续训时是"本阶段再练多少步"，不是累计目标。**
- 根因：SB3 在 `reset_num_timesteps=False` 时执行 `total_timesteps += self.num_timesteps`
  （`base_class.py:416`）。
- 影响：按累计算的话每级都会被放大成几百万步，课程推不到最后一级（第一次跑课程就这么废掉的：
  想练 25 万步，实际练了 100 万）。

**A4 —— `gradient_steps` 的语义不是"每次更新几步"。**
- 真相：它是"每收集一轮做几次更新"，而**一轮 = `train_freq × n_envs` 个环境步**
  （`collect_rollouts` 里 `self.num_timesteps += env.num_envs`）。
- 实测旧 checkpoint：`_n_updates=412,498 / num_timesteps=3,300,024 = 0.1250`——
  写 `1` 配 8 环境，UTD 只有 **1/8**，330 万环境步只做了 41 万次更新。
- `-1` 是 SB3 的标准 UTD=1（`gradient_steps = rollout.episode_timesteps = 8`），
  代价是墙钟时间约 8 倍。
- ⚠️ **改成 UTD=1 到底有没有好处还没测出结论**。实测 UTD=1 反而让策略塌得更快
  （rw4 用 1/8 撑了 40k 步，rw5 用 1 只撑了 20k），但那几轮都叠着别的变量。
  **不要把它当成"已知更优"来推荐。** 历史课程一直用的是 1。

**A5 —— TensorBoard 全写进同一个 `runs/`。**
- 根因：SB3 只按 `SAC_<n>` 递增编号，几十轮课程下来一个目录里堆几十个 event 文件、
  步数区间互相重叠。
- 后果：读曲线时会把不同学习率、不同地形的 run 混成一条（实测被误导过一次：
  3.35M 处同时出现 −11.98 / 122.5 / 50.61 / 219.4 四个"同一个点"）。
- 修法：按 `--save-dir` 的名字分目录（`train.py` 里已做）。

### B. 奖励设计类

**B1 —— "站着不动"必须是**恒等于** 0 分。这是全套奖励唯一必须守住的不变量。**
调了三轮才修干净，每一轮都是同一个病根的不同表现：

- 第一轮：`base_height` 写成 `0.5*exp(-err²/0.01)` 奖励、`track_sigma=0.25` →
  **零动作站着不动白拿 +674 分/回合**（离地高度 0.50/步 + 速度跟踪 0.20/步），
  而 `progress` 是 −0.0001。SAC 几步就找到这个零风险盆地，1e6 步后平均 vx 只有指令的 1/10、
  脚最多抬 7 cm（上不了 8 cm 的槛），回报却比真去爬还高（790 vs 456）。
  → 改成偏差惩罚 `max(0, |err| − 0.05)`（站对了正好 0）。
- 第二轮：`track_sigma` 收到 0.1 只治了一半。高斯速度跟踪在 vx=0 时的取值是 `exp(-cmd²/σ²)`：
  指令 0.8 时 0.002，但**指令 0.4 时是 0.19/步**——而指令在 (0.4, 0.8) 均匀采样，低指令那一半没被覆盖。
  零动作站满 1000 步仍有 **+148.7 分**。三个状态的回报排序整个反了：
  爬上去卡住 **−125** < 走到槛前停住 **−1.6** < 站着不动 **+148.7**。
  → 修法是**减掉 vx=0 的地板**：`max(0, exp(-(vx−cmd)²/σ²) − exp(-cmd²/σ²))`，
  vx=0 精确为 0 且与 σ、cmd 都无关。
- 第三轮：`progress` 本身是**年金**（每步都付、付两百多步）。实测 5 个回合的分项：
  爬完 6 级的回合总回报 +321，其中 `progress` +258.2、而爬楼专项收入只有 55 分
  （`climb` +5.1、`level_bonus` +30、`success` +20）——**爬完整段楼梯不如在平地快走一段**。
  → 改成**势能** `ΔΦ`（`Φ = x`，走廊内、`x_cap` 之前），权重 2.0 → 100.0（单位从"分/步"变成"分/米"）。

**B2 —— 势能塑形里的 γ 是陷阱。**
- 教科书形式 `γΦ(s') − Φ(s)` 只在 Φ **有界**时安全。`Φ = x` 沿路一直增长，折扣就变成
  **按停留时长计的税**：实测 50 Hz 下 Φ≈1.3、狗在**往前走**（Δx = 0.0008 m/步）时单步仍是
  `0.995×1.2717 − 1.2725 = −0.0071`，因为泄漏项 `(1−γ)Φ = 0.0064` 比它大一个数量级。
  要打平得每步走 `0.005·Φ` 米 → Φ=3.5 时要 0.875 m/s，**比指令速度上限还高**。
- 看到 `progress = −664` 时不要以为是 bug，就是这个。站满 1000 步白扣 650 分。
- **修法：直接用 `ΔΦ`**（`_progress` / `_climb` 都不乘 γ，`climb_gamma = 1.0`）。
  展开后精确 telescope、站着恒为 0、来回走净值 0，刷不了分。

**B3 —— 走廊权重 `w(y)` 必须乘在"增量"上，不能乘在"累积势能"上。**
- 第一版是硬闸门（`|y|>1.25` 时 Φ 直接置 0）：单步在 x=3.0 处横跨闸门就是 ±300 的悬崖，
  实测打出 `progress = −376 / −664 / −654`，看着像"它在倒退 6 米"（场地一共才 4 m 宽）。
- 第二版改成连续衰减 `1 − (|y|/gate)³`：抹掉了单步跳变，**抹不掉结算**——
  一个跑到 x=3.83、横漂到 |y|=1.42 的回合，w 归零，前面攒的势能全部吐回去（progress = −174），
  明明净前进了 2.2 m。后果：去爬然后摔变成 −211，站着不动只有 −52，**"站着不动"重新变成最优解**。
- 终版 `progress = Δx · w(y)`：走廊外 Δx 挣 0 分（照样防住绕旁边平地白拿），
  但漂出去不倒扣已经挣到的；走廊内仍然 telescoping。

**B4 —— 惩罚项要留死区和底噪，否则它们惩罚的是"正在爬"这件事。**
爬台阶时狗必然是"前脚在上、后脚在下"的跨坐姿势，机身天然比目标矮一截；
小腿顶着台阶沿往上蹭时接触力稳定在几十牛——那是爬不是撞。
原样计费时"卡在槛上"整回合要赔 125 分，策略于是学会**根本不靠近台阶**
（实测走到 x≈0.94、机身前缘正好顶在 x=1.1 的槛面上就停住，20 万步毫无进展）。
→ `base_height` 留 0.05 m 死区、`collision` 减 25 N 底噪。

**B5 —— 前进项必须在 `|y| > 1.25`、`x > 3.5` 处归零。**
地形各方块只有 y∈[−2, 2] 宽，外面是**无限平地**。不设闸门的话，绕到地形旁边走平地
就能白拿前进奖励（x 无上限地涨）——这是这套地形里最容易刷的分。

**B6 —— 预热数据必须重新标注奖励（`POSTURE_TERMS`）。**
- 现象：奖励改完之后，预热 buffer 的平均分变成 **−0.286 分/步**（A=1.0、8 回合、8/8 摔倒）。
  分项：`yaw_rate −0.181`、`yaw −0.162`、`lateral −0.142`，而 `progress +0.156`、`track_lin_vel +0.150`。
- 根因：`trot_action` **只驱动大腿和小腿**（`a[1::3]`、`a[2::3]`），**没有髋关节**，
  横向和航向物理上不可控、必然自转横漂。而新奖励把"站着不动"砍到**恰好 0**，
  于是预热在教 critic：**"小跳比站着差"**——与它的设计意图完全相反。
  （旧奖励下站着有 +0.19 保底，所以这个反转一直没暴露。）
- 修法：`collect_seed_transitions` 里把姿态/能耗项
  （`base_height lateral orientation yaw yaw_rate action_rate torques collision`）置零，
  只保留任务项。置零后 **+0.238 分/步**（小跳段 +0.283 vs 站立 0）。
- 这仍只是 critic 的**起点**，之后会被真实转移逐步纠正，不影响正确性。

### C. 仿真 / 几何类

**C1 —— 出生点必须按整只狗的脚印取最大地形高度，不能用身体中心那一点。**
用 `TerrainHeight.top_footprint`，不用 `TerrainHeight.top`。
- 根因：狗身长约 0.4 m，横跨高度不连续处（槛的棱、台阶的立面）时，**中心还在平地、
  前脚已经探到台阶上方**。按中心高度出生会把前脚连同小腿**直接生成在地形内部**。
- 实测：scale=0.62 时 24 个出生点里 **17 个有脚被埋，最深 0.0915 m**；求解器一上来就给巨大
  接触力，狗在 0.3 秒内被弹翻。**穿深 ≈ 那条棱的高度差，所以地形越高越严重**
  （scale 0.35 时只有 3 cm，到 1.0 就是 15 cm）。
- 这是"课程越往上、训练反而越差"的一个真凶，**和 reward、学习率、action_scale 都无关**。
- 顺带：`home` 关键帧的脚底本来就陷进地面约 18 mm，reset 时抬到地形上方让它自己落稳是对的。

**C2 —— `reset` 后必须 `mj_forward` 一次。**
否则接触/传感器数据还是上一回合的。**`geom_xpos`（四只脚的位置，支撑面高度从它取）同理**：
它只在前推/步进里更新，`prev_terrain_h` 必须在 `mj_forward` **之后**取，否则读到的
是上一回合的落点——出生点离台阶近时能差一整级。`core_parity` 的 `_pose()` 也要跟着前推，
不然合成状态量的是"别人脚下那片地形"。

**C3 —— 摔倒判据要有余量，不能凭直觉取。**
完全瘫倒时 base 离地约 0.077 m，而跨台阶（机身已在上一级正上方、脚还在下一级）余量最低
约 0.12 m。所以 `fall_clearance = 0.10`——**取 0.12 以上会把正常的跨步误判成摔倒**。
`flip_rad = 1.0`：跨 0.15 m 台阶时 pitch 合法地能到约 0.4 rad，取 0.8 太紧。

**C4 —— `goal_z` 必须按地形量出来，不能写死。**
`goal_z = terrain.top(goal_x) + goal_clearance`。写死 1.05 在课程缩放后失效
（scale=0.3 的平台只有 0.28 m 高，永远判不了成功）。
没有台阶的地形（`flat`/`steps`）必须把 `goal_z` 设成 `inf`——否则 `flat` 上"走到 x≥3.3"
就判成功、回合提前结束（实测平地回报从 715 掉到 281）。

**C5 —— 抬脚高度必须在**跳过落地过程之后**量。**
reset 把狗放在地形上方 0.30~0.33 m 让它自己落稳，**下落时四条腿全是悬空的**，
那一瞬间量到的"抬脚高度"能到 0.15 m——比真走路高一倍，会得出"脚抬得起来"的错误结论。
`diag lift` 的 `skip` 参数就是为此加的。

**C6 —— "当地地形高度"必须取自四只脚（`support_height`），不能取机身中心处的 `top()`。**
C1 说的是出生点，这里说的是奖励与终止用的那个高度——同一个"地形在棱处不连续"的两面。
`climb` / `level_bonus` / `base_height` / `is_fallen` **全部**吃这个数，取错一处、四处都错。
- 根因：机身中心越过台阶立面时，狗**还在平地上、脚一个都没上去**，`top()` 已经返回上一级的高度。
- 实测（s0.86 模型，台阶 0.1462 m）：`climb` 一次白拿 0.1462×100 = **+14.6**、`level_bonus` **+5.0**，
  而当时的姿态是四只脚里只有一只搭在台阶上、roll 已经 −0.66 在侧翻。10 个评估回合里 7 个
  在 25~59 步内以 roll≈−1.0 结束——**"伸头蹭一下台阶就倒地"是这套奖励的最优解**。
- 这也解释了"最远 x 卡在 2.1 且对课程高度不敏感"：2.1 是几何位置（第一级台阶立面），
  不是难度问题。真让前脚爬上台阶（x 更靠前、脚已踩上第 1/2 级）拿到的分**一模一样**，
  等于没有任何梯度去完成"把后腿也收上来"那半截动作。
- 取法：四只脚下方地形高度的**中位数**。踏面深 0.19~0.28 m < 狗脚印 0.4 m，狗站楼梯上
  永远是两条腿在一级、两条腿在相邻一级，中位数正好落在**它踩着的那一级**：
  平地=0、两前腿上一级=半级、跨两级=较低那级、上了顶平台=顶面高度。
  取 min 要求四条腿全上去（踏面放不下，信号太稀疏）；取 max 一只脚悬空就能刷分，和取错一样。
- **torch 侧不能用 `torch.median`**：偶数个样本它返回中间偏下那个，而 `np.median` 取中间两个的
  平均，四只脚横跨两级时两者差半级台阶——而那恰恰是最常见的一帧（`core_parity` 有专门一条盯它）。
- 连带：`is_fallen` 的入参从"地形对象 + 位置"改成"支撑面高度 + 位置"；诊断里看"最高台阶"
  要用 `env.support_level()`，用 `terrain.level(x, y)` 会报出比实际高一级的数
  （就是当年那个虚高的来源）。
- **Isaac 侧多一个接线坑**：`FL_foot` 是个**空 body**，位置正好压在脚底球心上
  （calf 系下 `0 0 -0.213`）。转 USD 时它可能被并进 `*_calf`，那样 `body_pos_w` 拿到的
  是膝关节、采样点整体偏 0.21 m——不崩不报错。`smoke.py` 的抽查里有一条量它
  （站住时脚底球心离地 ≈0.022，膝关节 ≈0.235）。

### D. 评估 / 选模型类

**D1 —— `model_of` 必须取每级最好的模型，不能取最后一个。**
难度一顶到天花板，训练就开始把策略往"别冒险"上拽（摔倒会终止回合、丢掉剩下九成的回报），
于是**最后一个 checkpoint 恰恰是最差的**。
实测 `st_s056`：`model.zip` 回报 111 / 成功率 1/5，而 `best_model.zip` 是 269 / 4/5。
取最后一个等于每级都拿退化模型喂下一级，越滚越糟。
推论：如果某一级训练从没赢过它的起点，`best_model` 会**原样继承**上一级的模型。
拿两个阶段的 best 对比就能看出来——实测 `st2_s056` 和 `st2_s062` 的每维 `|mu|` 逐位相同，
它们就是同一个模型，**整条课程从 scale 0.56 起就在空转**。

**D2 —— `eval_episodes` 必须够（默认 20），否则 `best_model` 是在按噪声选。**
5 个回合的成功率标准误是 **±22 个百分点**，而 `best_model` 是在所有评估里**取极大值**——
选出来的必然是幸运样本。实测同一个模型：

| 评估回合数 | 报出的成功率 |
|---|---|
| 5（旧默认） | 4/5 = **80%** |
| 20 | 8/20 = 40% |
| 40 | 12/40 = **30%** |

这直接制造了"每一级训练都把模型练坏了"的假象：拿一个幸运样本的旧评估去比一个新评估，
必然下降。**看到"训练把模型练坏了"先查评估回合数。**

**D3 —— 行为是双峰的，看"到顶几个"而不是"平均末级"。**
要么爬完 6 级，要么很早就摔（同一个模型 20 个回合：确定性 9/20、随机 7/20）。
"平均末级"会被双峰分布稀释。

**D4 —— `eval/vx_ratio` 会在策略崩溃时静默变成 NaN。**
`np.mean([])` 的返回值：5 个回合全部在 100 步内结束时，装速度的列表是空的。
那个 NaN 看着像"仿真数值发散"，实际含义是"策略一上去就摔"。
`train.py` 里已在空列表时退化到全程平均并加 ⚠ 标注。

**D5 —— 评估要看 `eval/vx_ratio`（实际速度/指令速度），不要只看回报。**
回报有保底项时完全没有分辨力：曾经那个"原地站着"的策略回报 771，
比它真去爬槛的 456 还高。

**但要知道这个数是怎么算的**：`EvalMetricsCallback` 只统计 `step_count > 100` 的步
（本意是跳过起步那 2 s 的加减速）。回合一短，这个窗口就只剩**尾巴**——而尾巴正好是
"走到台阶前停住不动"那段，于是 `vx_ratio` 打印出 0.1× 这种数，看着像"根本不会走"，
其实前面 1.8 m 是好好走过去的（`mean_max_x` 从出生点 0.2 一路到 2.0）。
要看"走没走起来"**同时**看 `mean_max_x` 和回合长度；`vx_ratio` 只在回合接近 1000 步
（真的走满 20 s）时才是稳态速度。

### E. 工具 / 操作类（agent 自身要小心）

**E1 —— `pkill -f 'go2_sac.train'` 会杀掉自己。**
`pkill -f` 匹配的是**完整命令行**，而你的 bash 命令行里就含这个字面量。
✅ 用 `pkill -f 'go2_sac[.]train'`（方括号让模式不匹配自身），
并且**把它单独放一条命令**，不要和任何含该字面量的命令写在同一个 `bash -c` 里。

**E2 —— Bash 工具默认超时 120 s。** `sleep 540` 之类必须显式传 `timeout`（毫秒，上限 600000），
或者用 `run_in_background`。

**E3 —— 不要在共享 scratch（如 `/tmp`）里用通配符批量删除。**
`rm -rf /tmp/lrtest_*` 这类命令会被权限系统拦（"Shared Scratch Sweep"）。
用**新的、具名的目录**，需要清理就逐个删。

**E4 —— SB3 内部 API 的几个坑（写诊断脚本时会踩）。**
- `m.policy.actor.get_action_dist_params(obs)` 返回 **3 个值**（SB3 2.9），不是 2 个也不是 4 个。
- **不要在 `torch.no_grad()` 里算 `mu/log_std` 然后拿去求导**——张量没有 `grad_fn`，
  报错是 `element 0 of tensors does not require grad`。
- 同一组参数连续 backward 两次会报 `Trying to backward through the graph a second time`，
  要么 `retain_graph=True`，要么每次重新前向。
- `m.policy.actor.log_std` **是网络层不是张量**（`.detach()` 会报
  `'Linear' object has no attribute 'detach'`），要读 σ 用 `get_action_dist_params`。
- `m.policy.actor(obs)` **直接返回动作**，不是 tuple，`[0]` 取到的是第一行。
- `float(m.ent_coef)` 会报 `could not convert string to float: 'auto'`，
  用 `float(m.log_ent_coef.exp())`。
- 读 TensorBoard 的 `ScalarEvent`：`for e in scalars: e.value`，
  `for _, v in scalars` 会报 `cannot unpack non-iterable ScalarEvent`。

**E5 —— `play.py` 退出时 `glfwTerminate()` 会段错误。**
本机 Wayland 下 glfw 的清理函数必崩，**与本项目代码无关**。`play.py` 已经绕过：
viewer 跑完不调 `glfwTerminate`，主动按正常退出码结束进程。
看到退出时那段 segfault 不要当成仿真或策略的问题去查。

**E6 —— `ctrl/go2_ctrl.py` 的构造函数会读 `sys.argv[1]` 当网卡名。**
原始写法是 `ChannelFactoryInitialize(0, sys.argv[1])` 直接取命令行第一个参数。
于是 `play.py --model foo.zip` 一导入它就会执行 `ChannelFactoryInitialize(0, "foo.zip")`。
已给构造函数加可选 `domain_id` / `interface` 参数，**显式传入时优先于 `sys.argv`**，
不传时行为不变。凡是从别的脚本（而不是直接 `python3 ctrl_test.py 网卡名`）导入 `Go2Ctrl`，
都必须显式传这两个参数，不要依赖 `sys.argv`。

### F. PPO / on-policy 专有（SAC 那份代码里没有这些坑）

**F1 —— 想改 `n_steps` / `batch_size`，绝不能调 `model._setup_model()`。**
- 现象：它会**重建策略网络**。`OnPolicyAlgorithm._setup_model()` 里有一句无条件的
  `self.policy = self.policy_class(...)`，跑完权重全变成随机初始化的。
- 根因：SAC 是 off-policy，`_setup_model` 只管网络和 buffer；PPO 复用了同一个函数，
  而 PPO 的策略网络也在这个函数里建。
- 修法（`go2_ppo/train.py` 续训段）：**只重建 rollout buffer**，照抄 `_setup_model` 里建 buffer 的
  那几行（`rollout_buffer_class(n_steps, obs_space, act_space, device, gamma, gae_lambda, n_envs)`），
  并且放在 gamma/gae_lambda 都设好之后。
- 为什么安全：`rollout_buffer` 在 `_excluded_save_params` 里，**根本不存进 checkpoint**，
  `load()` 时本来就会重建，所以重建它不丢任何东西。
- 实测：续训（`--n-steps 256→128 --batch-size 256→64`）前后 `policy.state_dict()` 的
  sha256 **逐位相同**（`ac4f1709…`），buffer 变成 `(128, 2, 45)`。

**F2 —— `clip_range` 和 `learning_rate` 从 `load()` 回来时**形态不一样。
- `clip_range` 是 `FloatSchedule`（`PPO._setup_model` 无条件包一层），而 `PPO.train()` 里写的是
  `self.clip_range(self._current_progress_remaining)`。直接 `setattr(model, "clip_range", 0.2)`
  → 第一次更新就抛 **`TypeError: 'float' object is not callable`**。
- `learning_rate` 反过来：`BaseAlgorithm.__init__` 只是把它存下来，load 回来**就是裸 float**，
  所以 `model.learning_rate(1.0)` 会同样报 `'float' object is not callable`（**实测踩到过**，
  当时的写法是假设它俩都是 schedule）。但它仍然要连 `model.lr_schedule` 一起换（同 A1）。
- 修法：用 `_lr_or_clip_value(v) = float(v(1.0)) if callable(v) else float(v)` 统一取值，
  `clip_range` 用 `FloatSchedule(...)` 重新包一层。

**F3 —— `log_interval` 在 on-policy 里是"训练迭代数"，不是回合数。**
SAC 那份的 `log_interval=4` 是"每 4 个结束的回合记一次"；PPO 的默认 `1` 是
**每采完一轮 rollout（`n_steps × n_envs` 步）就记一次**。别把两个数字互相搬。

**F4 —— `train/*` 曲线比 `rollout/*` 晚一个记录点，只跑一轮时压根没有 `train/*`。**
- 根因：`OnPolicyAlgorithm.learn()` 里 `dump_logs()` 在 `self.train()` **之前**调用，
  而 `logger.dump()` 才是真正写 TB 的时刻。于是第 N 轮更新算出来的 `train/approx_kl` 等，
  要到第 N+1 次 dump 才落盘（记在第 N+1 轮的步数上）。
- 实测：512 步（1 轮迭代）跑完，TB 里只有 11 个 `rollout/* / eval/* / time/*` 标签，
  **一个 `train/*` 都没有**；1024 步（2 轮）才有，且全部落在 `step=1024` 这一个点上。
- 推论：**看到"`train/*` 怎么没数据"先看是不是跑得太短，不要怀疑 TB 写坏了。**
  另外最后一轮更新的 `train/*` 不会出现在 TB 里。

**F5 —— PPO 的 `n_steps` 不能照抄 SB3 的默认 2048。**
策略 50 Hz、回合上限 20 s = 1000 步，2048 步是 **41 s**——一轮 rollout 比整个回合还长，
同一段轨迹的陈旧数据被反复拿来做优势估计。默认改成 **256**（= 每环境 5.12 s）。
`batch_size=256` 取 `n_steps × n_envs` 的约数，换 `--n-envs` 也不会出现零头 minibatch。

---

### G. Isaac Sim 迁移（`go2_issac/`）

**G0 —— 当前开发机跑不了 Isaac Sim，这不是"配置问题"是硬件。**
`nvidia-smi` 不存在、只有 AMD HawkPoint1 核显、内存 27 GB（要 ≥32）、
`/` 剩 21 GB（要 ~50）、Python 3.10.12（要 3.11）。**Isaac Sim 没有 CPU 回退**——
`--cpu` 只是"用 CPU 跑张量"，物理和渲染照样要 GPU。别在这台机器上试装。
所以 `go2_issac/` 刻意分两层：语义层（本机可证明）+ 接线层（只能上机）。
这是没有显卡时唯一能把风险压下去的办法，**不要为了"统一"把语义层搬进 `mdp/`**——
那样它就跟着 isaaclab 一起变成不可测的了。

**G1 —— 增益不在 USD 里，MJCF 转换器不管这件事。**
`default_drive_stiffness` 是 **URDF** 转换器的字段，MJCF 那边没有。
`go2.xml` 的 `<motor>` 只变成 USD 的 DriveAPI，kp/kd 必须由 `ImplicitActuatorCfg` 在
实例化时写回。而且 `damping` 要写 **3.6 不是 kd=3.5**：MuJoCo 的 12 个铰链自己带
`damping=0.1`，Isaac 的隐式执行器只有 `damping` 一个旋钮，两个要加起来
（依据：`assets/go2_reference.json` 的 `joints[].damping`，实测 0.1）。

**G2 —— 关节顺序有三个，只认名字不认下标。**
MJCF 的 `qpos` 跟身体树（FL,FR,RL,RR）、执行器/DDS 是 FR,FL,RR,RL、USD 里是导入顺序。
`mdp/state.py` 里 `JOINT_NAMES` 显式写死成 DDS 顺序，然后按名字查下标，启动时断言 12 个
名字全都对得上。**动作也要按名字重排一次**（`raw_action`）：动作项内部的顺序来自
`find_joints`（= USD 顺序），和 `State.q/dq/tau` 用的 DDS 顺序不一定一样，错了不报错、
只是学不出来。同理不依赖 `preserve_order` 这个字段（老版本没有）。

**G3 —— `prev_action` 必须是"夹过之后"的动作。**
`env.py:194` 是 `action = clip(action); self.prev_action = action`——先夹再存。
Isaac 的 `ActionTerm.raw_actions` 是"网络吐出来的原值"，夹取只改 `processed_actions`。
所以 `state.raw_action()` 里**自己再夹一次**。这不是边角情况：策略初始 σ=1.0，
约 1/3 的动作 `|a|>1`，不夹的话观测里的"上一拍动作"和 MuJoCo 侧系统性对不上。

**G4 —— 地形的坐标偏移只能上机校准，而且只影响一个常量。**
Isaac Lab 把 patch 的**中心**写进 `env.scene.env_origins`，但"中心"这个说法在版本之间
变过。做法不是猜，是 `smoke.py --check-terrain` 反推：对不上只改
`course.py:COURSE_ORIGIN_FROM_ENV_ORIGIN`。同理高度场的 `horizontal_scale`/`vertical_scale`
也要验（0.7 档的台阶高度恰好是 `vertical_scale` 的整数倍，量化误差 ≤0.5 mm）。

**G5 —— 跨仿真器不能续训，这是物理决定的。**
MuJoCo 是 `elliptic` 摩擦锥 + `impratio=100`、脚底 `condim=6`（带扭转/滚动摩擦）、
`priority=1` 独占接触对；PhysX 是各向同性库仑摩擦、没有 priority。
**是两个不同的物理**，不是同一个物理的两种实现。所以 `go2_ppo` 的 `.zip` 与 Isaac 的 `.pt`
互不通用（`train.py --resume` 会直接拒绝 `.zip`），曲线只能比趋势不能比数值。
`go2_common/` 那份代码仍然只有一份——变的是仿真器，不是任务定义。

**G6 —— rsl_rl 的超参名和 SB3 全不一样，映射表是唯一来源。**
见 `agents/rsl_rl_ppo_cfg.py` 的模块注释。几个**语义**上等价但容易记错的：
`gae_lambda`→`lam`、`ent_coef`→`entropy_coef`、`batch_size`→`num_mini_batches`（条数→份数）、
`log_std_init=0` ⇔ `init_noise_std=1.0`、`target_kl` 早停 ⇔ `schedule="adaptive"+desired_kl`。
另外 rsl_rl 的 `use_clipped_value_loss` **默认是 True**，而这边一直是"不裁 value"，
所以必须显式关掉（默认真空两边相反，最容易静默出错）。

**G7 —— 超时走 `truncated` 这件事在 Isaac 里叫 `time_out=True`。**
终止项的**名字**必须是 `time_out` 且带 `time_out=True`，rsl_rl 才会把它填进
`extras["time_outs"]` 做 bootstrap。写成普通终止项 = 超时处价值被截断，和 `env.py` 的
`truncated` 语义相反（第 4 节不变量 8）。

---

## 6. 实测数据（回答"这机器人能做多难"）

### 6.1 `action_scale` 是几何约束，不是调参

关节目标 = 默认站姿 + `action_scale` × 动作，它直接决定狗能跨多高的坎。

**不要用"固定机身扫关节可达集"来估这个值**——那样量到的是"原地摆姿势脚能抬多高"，
跟"走着撞上坎、要靠摩擦顶上去"完全是两回事，会**严重高估**：
0.40 时静态可达 15.8 cm，听起来 8 cm 的坎绰绰有余，实际连碰都碰不上去。

可信的测法只有一个：拿手调开环小跳（无反馈，`train.trot_action`）直接撞坎
（`diag frontier`）：

| action_scale | 0.06 m | 0.08 m | 0.10 m | 0.12 m |
|---|---|---|---|---|
| 0.40 | 卡 | 卡 | 卡 | 卡 |
| 0.60 | 过 | 卡 | 卡 | 卡 |
| 0.80 | 过 | 过 | 过 | 卡 |

带反馈的策略比这套开环小跑大约好一档（0.40 时开环只能过 0.04，平地训出来的策略能过 0.06）。
场景里那两个坎是 0.08 m，所以 **A=0.4 根本过不去**：策略会走到 x≈0.94 就停住，
20 万步零进展。**这不是奖励或探索的问题**，当时却一直在奖励函数里找原因。

`diag lift` 直接量抬脚高度（脚是半径 0.022 的球，下表是**脚底**离地；
行是 action_scale、列是动作幅度）：

| | 幅度 0.5 | 0.7 | 1.0（接近饱和） |
|---|---|---|---|
| A=0.6 前脚 | 0.053 | 0.086 | 0.098~0.139 |
| A=0.6 后脚 | 0.034 | 0.065 | 0.104~0.128 |
| **A=0.8 前脚** | 0.077 | 0.103~0.124 | 开环撑不住 |
| **A=0.8 后脚** | 0.059 | **0.100~0.113** | 开环撑不住 |

要点：(1) A=0.6 时后脚要顶到接近饱和才过得了 0.08——0.7 幅时后脚底只有 0.065，还差 1.5 cm；
(2) 幅度拉满会把**开环**小跑跑摔，但不等于闭环策略不能这么用。

### 6.2 已训策略的真实上限 ≈ 0.10~0.11 m

用 A=0.7 训出来的策略（`st2_s056_full/best`，20 回合），横扫 terrain_scale：

| terrain_scale | 首级台阶高 | 成功/20 | 最高级 |
|---|---|---|---|
| 0.56 | 0.095 m | 8/20 (40%) | 6/6 |
| 0.60 | 0.102 m | 7/20 | 6/6 |
| 0.62 | 0.105 m | 5/20 | 6/6 |
| 0.65 | 0.111 m | 2/20 | 6/6 |
| 0.70 | 0.119 m | 1/20 | 6/6 |
| 0.80 | 0.136 m | 0/20 | 4/6 |
| 1.00 | 0.170 m | 0/20 | 1/6 |

**是平滑退化，不是悬崖**——所以"某一级训练失败/成功"的判断在 5 回合评估下完全不可靠（见 D2）。

同一策略横扫 action_scale（不重新训练）：

| action_scale | terrain 0.56 成功/20 | terrain 1.00 成功/20 |
|---|---|---|
| 0.5 | 0/20（最高 1/6） | — |
| **0.7（训练值）** | **8/20** | 0/20（1/6） |
| 0.9 | 4/20 | 0/20（1/6） |
| 1.1 | 1/20 | 0/20（1/6） |
| 1.3 | 0/20（3/6） | 0/20（1/6） |

结论：**策略死死焊在它被训练的那个 action_scale 上**，调大调小都变差；
而在真实地形上任何振幅都是 0/20、只到 1/6 级、停在 x≈2.2。
**振幅调不动这个上限，要更高必须重新训一套步态。**

### 6.3 地形高度课程的分级

`run_stairs.sh` 的梯子（`--terrain-scale` → 首级台阶高）：

```
0.56 (0.095) → 0.62 (0.105) → 0.70 (0.119) → 0.78 (0.133)
→ 0.86 (0.146) → 0.93 (0.158) → 1.00 (0.170)
```

缩放只作用于几何体高度（`size[2]`/`pos[2]` 同乘，x/y 脚印不动），
所以台阶**踏面深度不变、只是变矮**——这正是想要的"变简单"。

### 6.4 预热（demonstration seeding）确实有效

SAC 配一个没有相位输入的前馈 MLP，要靠逐维白噪声"发现"周期步态；而自动熵系数会把噪声
压到 ±0.06 rad，能走起来的对角小跑需要 **±0.25 rad 的相干振荡**。结果 buffer 里全是"站着"，
Q 只学到"站着最好"，actor 永远不迈腿。
做法：先用 `trot_action` 采 3 万条转移填进 buffer（SACfD），**只给 Q 一个起点**。
实测 200k 步后平均 vx 从指令的 **0.00 倍变成 0.79 倍**。

配合 `log_std_init=0.0` / `target_entropy=-6.0`（不是 SB3 默认的 −1.0 / −12）：−1.0 时
α 会迅速衰减到 0.024、200k 步后 σ 只剩 0.25。

---

## 7. 当前状态与未解问题

**已完成**：平地 ✓、两个 0.08 m 槛 ✓、六级台阶（0.095 m/级）✓。

**PPO 通道**（`go2_ppo/`）：代码已建好，静态检查全过；跑过一次 512 步的空跑（采一轮 rollout、
做一次更新、评估回调正常、TB 有数），**但没有任何有意义的训练**。所以：
- 上面所有关于"能爬多高""怎么调超参"的结论**全部来自 SAC**，不能当成 PPO 的；
- PPO 没有预热，它的起点就是 σ=1.0 的随机策略（实测空跑时回合回报 −20.8、几乎原地就摔），
  能不能像 SAC 那样从零走起来，**没有数据**，要试就从平地那一级开始。

**Isaac Sim 通道**（`go2_issac/`）：**一次都没跑过**，因为这台机器没有 NVIDIA 显卡
（硬件级别的跑不了，见 §5-G0）。能做到的保证和不能做的保证都写清楚：

- **已经用真 MuJoCo 对拍验过的**（本机可跑，全绿）：`course.py` 的地形几何、
  `core.py` 的 45/46 维观测、14 项奖励、终止与成功判据——逐元素差别在 1e-15
  级（浮点求和顺序）。资产真值（质量 15.206408 kg、前后腿不同的关节限位、
  `damping=0.1`、脚底球 `priority=1`/`friction=0.4`）是从编译后的 `MjModel` 直接导出的。
- **完全没验过的**：PhysX 物理本身、Isaac API 的接线、训练能不能收敛。
  这三件事只能上机，`smoke.py --all` 就是为它们准备的（五节，一节一个结论）。
- 所以：**上机后如果 `smoke.py` 全绿而训练不收敛，那不是"代码写错了"，
  是这套接触动力学要不要重调奖励/超参的问题**——和当年在 MuJoCo 上从零开始面对的是同一类问题。

**没上去的**：**真实的 0.15~0.17 m 台阶**。现有 A=0.7 步态在任何振幅下都够不到（见 6.2）。

**A=1.0 大振幅从零重训：已做完，结论是失败**（`models/amp1_s035`，terrain_scale 0.35，
2M 步，`gradient_steps=1`，跑完 21:03–22:26）。

- 训练本身跑通了：750k 起 `max_level` 稳定 6/6，1.3–1.75M 步成功率在 55%~90% 之间。
- 但**最后 250k 步发散**：成功率在 0.90 / 0.20 / 0.90 / 0.10 之间震荡（mean_reward
  325 → 19 → 300 → 34）。最终 `model.zip` 只剩 **4/20**，而 `best_model.zip`（存于 1.0M）是
  **16/20**——**又是 D1：最后一个 checkpoint 恰恰是最差的，差 4 倍。**
- **横扫 terrain_scale，A=1.0 每一档都输给 A=0.7**（同脚本、同 20 个种子）：

  | scale | 首级高 | A=0.7（训于 0.56） | A=1.0（训于 0.35） |
  |---|---|---|---|
  | 0.35 | 0.059 | 14/20 | 16/20 |
  | 0.45 | 0.077 | **12/20** | 8/20 |
  | 0.56 | 0.095 | **10/20** | 2/20 |
  | 0.62 | 0.105 | **4/20**（6/6） | 0/20（2/6） |
  | 0.70 | 0.119 | **1/20**（6/6） | 0/20（1/6） |
  | 1.00 | 0.170 | 0/20（1/6） | 0/20（1/6） |

  在 0.35 那一格 A=0.7 从没训练过（它训于更高的 0.56）也拿到 14/20——A=1.0 多花的 200 万步
  没买到任何东西。腿摆动确实更大（0.23 vs 0.17），但**摆得高 ≠ 爬得上**。

**结论：`action_scale` 调不动上限，重新训一套大振幅步态也调不动。** 6.2 节测的是"策略焊死在
训练时的 action_scale 上"，这里是更强的一条：**换一套更大的振幅从头训，上限一样不动。**
不要再往这个方向投入算力。

**下一步（尚未验证的假设）**：A=0.7 那条课程是在出生点穿模 bug 下训出来的
（`st2_s056_full/best_model.zip` 存于 19:20，`top_footprint` 修复 20:10 才进 `env.py`），
而这个 bug 的穿深与地形高度成正比（0.35 档 3 cm、1.0 档 15 cm），**正好在它卡住的 0.62 往上最狠**。
它带着这个 bug 都能到 10/20@0.56——用修好的 env 重跑梯子，是当前最有依据的一次尝试。

**已证伪、不要再试的假设**（都做过测量）：
- ~~"奖励排序反了"~~ —— 同一批种子、同一套权重下：成功策略 +140.5/回合、半途摔 +53.3、
  更差 +1.4。排序是对的。
- ~~"critic 变悲观 → actor 跟着塌"~~ —— 把 `target_entropy` 从 −6 降到 −12 后，
  α 0.102→0.038、σ 0.411→0.252、Q 稳在 −5.91（起点 −4.92），**策略照样 20k 步塌到 0/20**。
  在 Q 完全健康的情况下行为崩掉，因果链不成立。
- ~~"critic 在策略动作附近没有坡度"~~ —— 实测 `|dQ/da| ≈ 5.1`，沿坡度走一小步 Q 涨 +0.220；
  且 `|g_Q| = 3.47 > |g_ent| = 2.49`，Q 项主导 actor 梯度。
- ~~"微调必然毁策略"~~ —— 0.62 那一级的策略其实爬得上去（6/6 到顶、25% 成功率），
  之前"失败"的结论是 5 回合评估比出来的假象。
- ~~"加幅度能提高攀爬上限"~~ —— 见上表，A=1.0 从头训完 2M 步，每一档都不如 A=0.7。

**仍未解释的**：在 0.56 档做微调，**20k 步就把 40% 的策略打到 0/20**，
在 UTD=1/8 和 UTD=1、旧奖励和新奖励、`target_entropy` −6 和 −12 下**都可复现**。
`sac` 里唯一始终没变过的共性是"warm start + 50 万条陈旧 buffer + 改了奖励函数"。

---

## 8. 在这个仓库里工作的纪律

1. **改任何奖励/超参之前，先读第 5 节对应的分类。** 这个仓库的绝大部分损失来自 A 类（静默失效）。
2. **宣称"X 改善了效果"之前必须有两组同种子、20 回合的对照。** 5 回合的差异全是噪声（D2）。
3. **一次只改一个变量**，并且把对照的数字写进注释。
4. **不要为了让指标好看而放宽成功判据或缩短评估**（第 3 节规范 7）。
5. **长时间训练放后台**（`nohup ... &` + `run_in_background`），
   但**不要并发跑两个训练**——16 核会被两个 8 环境 + 8 torch 线程的进程互相拖垮。
6. **训练完先看 `best/best_model.zip`，不是 `model.zip`**（D1）。
7. **报告失败要给出数字**。这个仓库的注释和 README 里到处是"实测 −376 / −664 / −654"这种
   原始数字——它们是唯一能防止重复踩坑的东西。
