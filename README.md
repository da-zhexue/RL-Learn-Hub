# 机器人强化学习运控
  
本仓库以宇树机器人与机器狗为例进行强化学习运控的学习。

## 环境
本仓库在以下环境验证通过（Ubuntu 22.04 + conda 虚拟环境 + ROS 2 Humble）：

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS (jammy)，内核 6.8.0-138-generic |
| ROS 2 | Humble |
| Python | 3.10.12（conda 环境 `sim`） |
| unitree_sdk2_python | 1.0.1 |
| unitree_ros2 | 0.3.0 |
| mujoco (pip) | 3.10.0 |
| cyclonedds (pip) | 0.10.2 |
| numpy | 1.26.4 |

## 宇树运行环境配置
下载仓库（本仓库所用python和ros2进行开发）
```
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git clone https://github.com/unitreerobotics/unitree_ros2.git
```
安装sdk（在虚拟环境下）
```
cd unitree_sdk2_python
pip3 install -e .

pip3 install mujoco
```
配置ros2环境（本仓库ros2版本为humble，如为foxy按unitree官方仓库说明进行配置）
```
sudo apt install ros-humble-rmw-cyclonedds-cpp ros-humble-rosidl-generator-dds-idl libyaml-cpp-dev
source /opt/ros/humble/setup.bash
colcon build
```
修改文件
```
nano ~/.bashrc
写入：export ROS_DOMAIN_ID=1
nano unitree_ros2/setup_local.sh
修改：source /opt/ros/humble/setup.bash
修改：source $HOME/unitree/unitree_ros2/install/setup.bash # 自己放仓库的路径
nano unitree_mujoco/simulate_python/config.py
修改：USE_JOYSTICK = 0 # 为1时需要外接手柄才能启动
```
测试
```
cd unitree_mujoco/simulate_python
python3 unitree_mujoco.py # 打开后所有电机失能，狗会趴下
# 另开一个终端
source unitree_ros2/setup_local.sh
cd unitree_mujoco/example/ros2
# colcon build # 仅执行一次
source install/setup.bash
./install/stand_go2/bin/stand_go2 # 狗会站起来一会后趴下
```

## GO2狗控制
完成Go2Ctrl类，有直接控制、插值平滑控制、读取当前姿态、重置函数。按照宇树官方说明，需要以500Hz进行控制，否则会失控，故开一个独立线程来以固定频率控制。经调试，电机kp大致取值50～60，kd取值3.5～4.0，过高会高频抖动。  
测试
```
cd unitree_mujoco/simulate_python
python3 unitree_mujoco.py
# 另开一个终端
cd ctrl
python3 ctrl_test.py
```

## GO2强化学习过地形（SAC）
用SAC训练Go2走过「平地 → 几个槛 → 几级台阶」。

### 地形
`unitree_mujoco/unitree_robots/go2/scene.xml`
| x 位置 | 内容 |
| --- | --- |
| -0.3 ~ 1.1 | 平地（出生点在这里） |
| 1.2 / 1.6 | 两个 0.08 m 高的槛 |
| 2.1 ~ 3.66 | 六级台阶，顶面依次 0.17 / 0.32 / 0.47 / 0.62 / 0.77 / 0.92 m |
| 3.14 ~ 3.66 | 顶平台（0.92 m），再往前是悬崖 |

台阶是**互相重叠**的（方块 pos_z 都是 0.02），所以每级**踏面只有约 0.19 m**；
而且**第一级从地面起就是 0.17 m**，比后面每级的 0.15 m 还高 2 cm。
x∈[1.7, 2.1] 是楼梯前仅有的 0.4 m 平路——出生点要落在这附近才有助跑。

### 目录
环境、地形、奖励都是 SAC / PPO 共用的，放在 `go2_common/`，两个算法包里只有各自的算法代码；
两个包的命令行参数、目录结构、`config.json` 格式完全一致，换算法只是换个模块名。

```
go2_common/config.py       常量、EnvCfg/RewardCfg 配置类、四元数工具
go2_common/terrain.py      课程变体、地形高度查询、成功/摔倒判据
go2_common/env.py          Go2TerrainEnv（Gymnasium 环境）
go2_common/reward.py       奖励函数：每项一个函数，权重在 RewardCfg 里同名对应
go2_common/train_utils.py  环境工厂、评估回调、从 config.json 还原环境配置
go2_sac/config.py          SAC 的 TrainCfg
go2_sac/train.py           SAC 训练入口
go2_sac/play.py            回放（viewer / dds）
go2_sac/diag.py            诊断工具（地形/站姿/抬脚/奖励分项/回合为什么结束）
go2_ppo/*.py               同上，把算法换成 PPO
```

### 训练
从头训：
```
python3 -m go2_sac.train --terrain flat --steps 200_000 --checkpoint-freq 40_000     # 先学会在平地小跑
```
然后跑地形课程（从上面那个平地模型出发，逐级放开地形）：
```
./run_curriculum.sh models/go2_sac_flat_xx/checkpoints/rl_600000_steps.zip
```
手工逐级等价于（`--steps` 是本级再练多少步）：
```
python3 -m go2_sac.train --terrain steps --terrain-scale 0.70 --fresh-reward \
    --steps 1_000_000 --resume models/go2_sac_flat_xx/...               # 5.6 cm 的槛
python3 -m go2_sac.train --terrain steps --terrain-scale 0.86 --fresh-reward \
    --reset-x -0.3 0.95 --steps 250_000 --resume models/cur_s070_steps/model.zip
python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --fresh-reward \
    --reset-x -0.3 0.95 --steps 1_000_000 --resume models/go2_sac_steps_s0.7_20261006_120850/checkpoints/rl_850000_steps.zip
python3 -m go2_sac.train --terrain full  --terrain-scale 0.70 --fresh-reward \
    --steps 1_000_000 --resume models/go2_sac_steps_20261006_143205/checkpoints/rl_1850000_steps.zip
python3 -m go2_sac.train --terrain full  --terrain-scale 0.85 --fresh-reward \
    --steps 1_000_000 --resume models/go2_sac_full_s0.7_20261007_101816/checkpoints/rl_2850000_steps.zip
python3 -m go2_sac.train --terrain full  --terrain-scale 1.00 --fresh-reward \
    --steps 1_000_000 --resume models/go2_sac_steps_20261006_143205/checkpoints/rl_1850000_steps.zip          
```
- `--resume` 接上一阶段：`.zip` 可省略，`checkpoints/rl_xxx` 这种路径也行
  （`config.json` 会自动往上找一层）；**换阶段必须显式写 `--terrain`**，
  否则沿用被续训模型自己的。
- `--steps` 续训时是**本阶段再练多少步**，不是累计目标。
- **换阶段必须加 `--fresh-reward`**，否则续训会把 checkpoint 里的旧奖励权重整个恢复回来。
- 从头训练会自动用小跑数据预热 replay buffer（不用管）；**续训默认不预热**（`--seed-trot 0`），要开就显式写。

### 回放
```
python3 -m go2_sac.play --mode viewer --model models/go2_sac_full_xxx/model.zip
```

### 奖励设置
共 14 项。每项在 `reward.py` 里是一个函数、在 `RewardCfg` 里是同名权重字段，一一对应；
正负号写在权重里，`compute_reward()` 加权求和。调权重只改 `RewardCfg`，不要改函数体。

**正向项（任务）**

| 项 | 形式 | 权重 | 说明 |
| --- | --- | --- | --- |
| progress | `Δx · w(y)`，走廊内的 x 势能增量 | +100 /米 | 主收入，走完全程约 +190 |
| climb | `Δ地形高度`（不是 Δ机身高度） | +100 /米 | 爬高，爬完六级 +57 |
| level_bonus | 登上新一级台阶 = 1，只涨不跌 | +5 | 离散里程碑，比连续势能好认 |
| track_lin_vel | `max(0, e^{-(vx-cmd)²/σ²} − e^{-cmd²/σ²})` | +1.0 | 步态质量（金额很小） |
| success | 站上顶平台 = 1（终止） | +20 | |

**惩罚项（姿态 / 平滑 / 能耗）**

| 项 | 形式 | 权重 | 说明 |
| --- | --- | --- | --- |
| yaw | `1−cos(yaw)` | −1.0 | 航向对准 |
| orientation | `(1−cos roll)+(1−cos pitch)` | −0.5 | 机身保持水平 |
| base_height | `max(0, \|z−h−0.27\| − 0.05)` | −0.3 | 站太低或太高 |
| lateral | `y²` | −0.3 | 别跑偏出地形 |
| yaw_rate | `ωz²` | −0.25 | 别原地打转 |
| action_rate | `‖aₜ−aₜ₋₁‖²` | −0.05 | 动作别抖 |
| collision | `max(0, 接触力−25N)` | −0.002 | 机身/大小腿撞地形，按力缩放、不终止 |
| torques | `Στ²` | −1e-4 | 省力矩 |
| fall | 摔倒/出界终止 = 1（成功不算） | −25 | 终止项 |

内部参数：`track_sigma 0.1`、`height_target 0.27`、`height_dead_zone 0.05`、
`collision_force_free 25 N`、`climb_gamma 1.0`、`progress_y_gate 1.25`、`progress_x_cap 3.5`。

**设置奖惩需要注意的事项**

1. **"站着不动"必须恒等于 0**：原先稳定站立奖励设为`+exp(-err²)`，在正确站立时会有最高奖励，后改为`-max(0, |z−h−0.27| − 0.05)`惩罚站立高度不对，否则狗卡在槛前不动的reward比尝试去跨过槛的还要高，根本不去尝试跨过槛。`track_lin_vel` 要减掉 vx=0 的地板：高斯跟踪在 vx=0 时是 `exp(-cmd²/σ²)`，指令 0.4 时高达 0.19/步。`progress` 用 Δx 势能，否则爬完整段楼梯的reward反而不如"在平地快走一段"。

2. **势能不能乘 γ。** 教科书形式 `γΦ(s')−Φ(s)` 只在 Φ **有界**时安全。虽然上台阶并非真无界，但若γ太低，`γΦ(s') − Φ(s) = γΔΦ − (1−γ)Φ(s)`当站在高处卡着无法再上前时，`(1−γ)Φ(s)`扣分超过了登上来的势能奖励，就会得出应该在平地上停着更优的结论。

3. **前进项必须在 `|y|>1.25`、`x>3.5` 处归零。** 地形各方块只有 y∈[−2,2] 宽，外面是**无限平地**——
   不设闸门的话，绕到地形旁边走平地就能白拿前进奖励，这是这套地形里最容易刷的分。

4. **加新项之前先问"它能不能被刷"。** 被否掉的例子：`feet_air_time`（不加速度指令闸门时，原地踏步和
   冲下悬崖都能刷）、用 `Δbase_z` 当爬升（抬屁股就能刷）、`exp(-err²)` 形式的姿态奖励（站对了白拿）。

### 预热
SAC 配一个没有相位输入的前馈 MLP，要从逐维白噪声里"发现"周期步态是做不到的：自动熵系数会把噪声压到
±0.06 rad，而能走起来的对角小跑需要 ±0.25 rad 的**相干**振荡。结果 replay buffer 里全是"站着"的转移，
Q 只学到"站着最好"，actor 永远不迈腿（实测平均 vx 是指令的 0.00 倍）。

做法是先用开环小跑（`train.trot_action`）采一批转移把 buffer 填上（SACfD），**只给 Q 一个起点**，
策略之后仍然自由学习。
- 从头训练**默认开**（`--seed-trot 30` + `--seed-rand 6`，不用管），采完立刻开始梯度更新，
  不等 `learning_starts`；**续训默认关**——预热喂的手调小跑比一个已经会走路的策略差得多
  （后腿抬不起来、没有航向控制），拿它当先验等于把好策略往回拽。
- 实测效果：200k 步后平均 vx 从指令的 **0.00 倍变成 0.79 倍**。
- **预热数据要重新标注奖励**（`POSTURE_TERMS` 置零）：开环小跳只驱动大腿和小腿、**没有髋关节**，
  横向和航向物理上不可控、必然自转横漂；姿态项原样计费会让这批数据变成 **−0.286 分/步**，
  而"站着不动"在新奖励下恰好是 0——预热反而教会 Q"小跳比站着差"，与意图完全相反。
  置零后是 **+0.238 分/步**（小跳段 +0.283 / 站立 0）。
- 小跑的幅度按**绝对关节摆幅**（0.24~0.40 rad）给、再除以 `action_scale` 折算成动作值，
  这样换 `action_scale` 时喂的还是同一个物理步态。直接写动作幅度（曾用 0.6~1.0）会随 `action_scale`
  一起放大——0.6 时摆幅冲到 0.6 rad，狗直接摔，采集量从 3.5 万条掉到 1.1 万条。

## GO2强化学习过地形（PPO）
和上面那节的**唯一区别是算法**：环境、地形、奖励、观测、课程分级、命令行参数、`config.json`
格式全都一样，`go2_common/` 里那份代码两边共用，只是模块名从 `go2_sac` 换成 `go2_ppo`。

### 训练
从头训：
```
python3 -m go2_ppo.train --terrain flat --steps 200_000 --checkpoint-freq 40_000      # 先学会在平地小跑
```
然后跑同样的地形课程（`--steps` 是本级再练多少步）：
```
python3 -m go2_ppo.train --terrain steps --terrain-scale 0.70 --fresh-reward \
    --steps 1_000_000 --resume models/go2_ppo_flat_xx/checkpoints/rl_200000_steps.zip   # 5.6 cm 的槛
python3 -m go2_ppo.train --terrain full --terrain-scale 1.00 --fresh-reward \
    --steps 1_000_000 --resume models/go2_ppo_steps_xx/checkpoints/rl_1000000_steps.zip # 再加六级台阶
```
`--resume`、`--fresh-reward`、`--steps`、`--terrain-scale`、`--action-scale` 的语义与 SAC 那节完全一致
（包括"换阶段必须显式写 `--terrain` 和 `--fresh-reward`"）。

### 回放
```
python3 -m go2_ppo.play --mode viewer --model models/go2_ppo_full_xxx/model.zip
python3 -m go2_ppo.play --mode dds    --model models/go2_ppo_full_xxx/model.zip   # 另开 unitree_mujoco.py
```

### 与 SAC 的差异
1. **没有预热**：PPO 是 on-policy，每轮采完就丢掉，没有 replay buffer 可以先填（命令行也没有
   `--seed-trot`）。PPO 也不太需要——它的初始探索噪声 σ=1.0，而 SAC 当年的死因是自动熵把 σ 压到 0.25。
2. **超参不同**：`n_steps=256`（SB3 默认的 2048 在 50 Hz 下是 41 s，**比整个回合 20 s 还长**）、
   `batch_size=256`、`n_epochs=5`、`ent_coef=0.005`、`target_kl=0.02`（approx_kl 超了就提前结束本轮更新）。
   `gamma=0.995` / 学习率 / `net_arch` 与 SAC 相同。可用
   `--n-steps --batch-size --n-epochs --ent-coef --target-kl` 临时改。
3. **不能跨算法续训**：`--resume` 只能接同一个算法存出来的模型。
4. **续训可以改 `--n-steps` / `--batch-size`**：rollout buffer 不存进 checkpoint，会按新值自动重建。

## GO2强化学习过地形（Isaac Sim）
把上面那套任务搬到 **NVIDIA Isaac Sim**：同样地形、同样 45 维观测、同样 14 项奖励、
同样课程分级，但算法换成 **rsl_rl**、资产用仓库里的 `go2.xml` **自己转 USD**。

> ⚠️ **当前这台开发机跑不了**：没有 NVIDIA 显卡（只有 AMD 核显），内存 27 GB、盘剩 21 GB，
> 都低于 Isaac Sim 的最低要求——而它**没有 CPU 回退**。所以这一节写的代码要搬到有卡的机器上跑，
> 步骤见 [go2_issac/RUNBOOK.md](go2_issac/RUNBOOK.md)。

为了在没有显卡的情况下也能保证正确性，代码刻意分成两层：

- **本机可证明的语义层**（纯 numpy / torch，不 import mujoco 也不 import isaaclab）：
  `course.py` 地形几何、`core.py` 观测/奖励/终止判据。它们和真的 MuJoCo 环境**逐位对拍**
  （`tests/course_parity.py`、`tests/core_parity.py`），差别在 1e-15（浮点求和顺序）。
- **只能上机验证的接线层**：`env_cfg.py`、`mdp/`、`agents/`、`train.py` / `play.py` / `smoke.py`。

所以上机后可能出错的只剩"Isaac API 接得对不对"，不会再有语义错误。

### 目录
```
go2_issac/course.py            地形几何唯一真源（8 个方块 + 高程图），纯 numpy
go2_issac/core.py              观测 / 14 项奖励 / 终止判据，纯 torch
go2_issac/env_cfg.py           CourseEnvCfg（ManagerBasedRLEnvCfg）+ 地形生成
go2_issac/mdp/                 rewards / observations / terminations / events / commands / metrics 薄壳
go2_issac/agents/rsl_rl_ppo_cfg.py  rsl_rl 超参 = go2_ppo/config.py 的逐项映射
go2_issac/convert_assets.py    go2.xml -> USD（在有卡的机器上跑）
go2_issac/smoke.py             上机自检：资产/关节顺序/地形坐标/观测奖励/零动作静置
go2_issac/train.py play.py     训练 / 回放
go2_issac/assets/*.py          从编译后的 MjModel 导出资产真值与静置真值（本机跑，产物进 git）
go2_issac/tests/*.py           和 MuJoCo 的逐位对拍（本机跑）
go2_issac/RUNBOOK.md           上机手册
```

### 和有卡机器的交接
本机能做的只有"把语义证明对"，剩下三步在那台机器上：

```
python3 go2_issac/convert_assets.py                       # 1. 转 USD
python3 go2_issac/smoke.py --all --num-envs 4             # 2. 自检（别跳）
python3 go2_issac/train.py --terrain flat --steps 200_000 --num-envs 512 --headless   # 3. 开训
```

### 与 MuJoCo 侧的差异
1. **算法库换成 rsl_rl**：全程 GPU、没有 Python 逐环境循环，吞吐高两个数量级。
   超参名全变了，映射表在 `agents/rsl_rl_ppo_cfg.py` 的模块注释里，改一边要照表同步。
2. **资产自己转**：用仓库里的 `unitree_mujoco/unitree_robots/go2/go2.xml`，
   **不用** Isaac Lab 官方给的 `UNITREE_GO2_CFG`。好处是整条链路可控、参数可追溯到
   `assets/go2_reference.json`；代价是驱动增益（kp=60 / damping=**3.6**）必须在
   `env_cfg.py` 里显式写回，见 RUNBOOK §2。
3. **跨仿真器不能续训**：PhysX 和 MuJoCo 的接触模型不同（摩擦锥、脚底 `condim`、
   `priority`），不是同一个物理的两种实现。`--resume` 会直接拒绝 `.zip`。
   曲线只能比趋势，不能比数值。
4. **没有独立的评估环境**：Isaac 里再起一个物理场景太贵，改成在训练环境里顺带统计
   （`mdp/metrics.py:EpisodeTracker`），口径和 `EvalMetricsCallback` 逐项对齐。