# 机器人强化学习运控
  
本仓库以宇树机器人与机器狗为例进行强化学习运控的学习。强化学习算法理论学习参考[Hands On Modern RL](https://walkinglabs.github.io/hands-on-modern-rl/preface/introduction)。

## 目标
主要目标为多种强化学习运控算法实践与Issac Sim的使用。
| 目标 | 完成 |
| --- | --- |
| SAC训狗上台阶 | ✅ |
| Issac Sim中PPO训狗 |  |
| Model Base的强化学习算法训狗 |  |
| WBC传统控制人形机器人跑步 |  |
| SAC人形机器人跑步 |  |
| PPO人形机器人跑步 |  |

## 本地环境
| 组件 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS (jammy) |
| ROS 2 | Humble |
| Python | 3.10.12（conda 环境） |
| unitree_sdk2_python | 1.0.1 |
| unitree_ros2 | 0.3.0 |
| mujoco (pip) | 3.10.0 |
| cyclonedds (pip) | 0.10.2 |
| numpy | 1.26.4 |

## 云端环境
| 组件 | 版本 |
| --- | --- |
| 显卡 | Nvidia RTX 4090 |
| 操作系统 | Ubuntu 24.04.5 LTS |
| Python | 3.11（conda 环境） |
| CUDA | 13.2 |
| Issac Sim | 5.1.0 |
| Issac Lab | 2.3.0 |

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

#### 参数

**SAC / PPO 共用**
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-h, --help` | — | 打印这份列表本身 |
| `--terrain {flat,steps,full}` | `full` | 课程阶段。`--resume` 且不给时，沿用模型自己的地形 |
| `--scene XML` | 自带的 `scene.xml` | 场景 xml 路径 |
| `--steps N` | `3_000_000` | 训练步数。从头训是总量，`--resume` 时是本级再练多少 |
| `--n-envs N` | `8` | 并行环境数。SAC 下买的是多样性不是速度；PPO 下一轮 rollout = `n_steps × n_envs` |
| `--seed N` | `0` | 随机种子 |
| `--save-dir DIR` | 见左 | 默认 `models/go2_sac_<地形>[_s<缩放>]_<时间戳>/`（PPO 则是 `go2_ppo_...`） |
| `--resume PATH` | 无 | 接着训，`.zip` 可省略；会读它同目录（找不到就往上一层找）的 `config.json`。PPO 只能续 PPO 自己的模型 |
| `--privileged` | 关 | 观测里加 1 维机身离地高度（真机上没有，仅作对照）：45 维 → 46 维 |
| `--reset-x LO HI` | `-0.3 0.3` | 起点 x 的随机范围，例如 `--reset-x 1.20 2.05` 直接练爬台阶 |
| `--terrain-scale S` | `1.0` | 地形高度整体缩放（课程用）：`1.0`=原场景、`0`=平地。x/y 脚印不变，只压矮高度 |
| `--action-scale A` | `0.6` | 关节目标幅度 `q_target = q_default + A·a`，决定能跨多高的槛。续训要么显式给、要么确认和上次一致 |
| `--learning-rate LR` | `3e-4` | 微调时调小（如 `1e-4`）。续训不显式给会被 checkpoint 的值静默覆盖 |
| `--fresh-reward` | 关 | 续训时不沿用 checkpoint 里的奖励权重，改用当前 `RewardCfg` 的默认值。换课程阶段必须加|
| `--eval-freq N` | `50_000` | 每 N 个环境步评估一次|
| `--eval-episodes N` | `20` | 每次评估跑几个回合。|
| `--checkpoint-freq N` | `200_000` | 每 N 个环境步存一个 checkpoint |
| `--torch-threads N` | `8` | 父进程 torch 的线程数（各子进程固定 1 线程跑物理） |

**SAC 专属**
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--target-entropy H` | `-6.0` | 目标熵。SB3 默认 `-12` 对 12 维动作过强，会把探索压没（见 AGENT.md §6.4） |
| `--gradient-steps G` | `-1` | 每收集一轮（= `train_freq × n_envs` = 8 个环境步）做几次梯度更新。`-1` = 标准的 UTD=1（见 AGENT.md §5-A4） |
| `--seed-trot N` | 看情况 | 用 N 条开环小跑轨迹预热 replay buffer；`0` = 关闭。从头训练默认开（30 条），续训默认关——预热喂的手调小跑比已经会走路的策略差得多（见下面[预热](#预热)一节） |

**PPO 专属**
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--n-steps N` | `256` | 每个环境每轮 rollout 采多少步。别用 SB3 默认的 2048：50 Hz 下那是 41 s，比整个回合（20 s）还长 |
| `--batch-size B` | `256` | minibatch 大小。建议取 `n_steps × n_envs` 的约数，否则最后一批是零头 |
| `--n-epochs E` | `5` | 每轮 rollout 的数据重复训几遍 |
| `--ent-coef C` | `0.005` | 熵奖励系数。PPO 初始 σ=1.0，熵不是主旋钮 |
| `--target-kl KL` | `0.02` | 早停阀：单次更新的 `approx_kl` 超过 1.5×它，本轮剩下的 epoch 全停 |

加了 `--resume` 之后，环境侧的`terrain / scene / reset_x / terrain_scale / action_scale / privileged` 和奖励权重会先整个从 checkpoint 同目录的 `config.json` 里恢复，命令行显式给了才覆盖（`--fresh-reward` 则把奖励权重换回当前默认值）；`--steps` 也从「总量」变成「本级再练多少」。

#### 预热
写死一个开环小跑的动作并录制数据进行训练，防止陷入卡住不动的局部最优解，让狗学会跑起来分可以更高。  
从头练默认预热，续练默认不预热，想要修改通过--seed-trot设置。

#### 课程
课程设置将原先较复杂的任务拆分为难度逐渐递增的任务，比如先学会平地跑，再学会上矮台阶，再学上高台阶，最后再跑完整任务。课程可以减少训练步数，降低落入局部最优解的可能。  
  
从头训：
```
bash sac_train.sh
```
或者等价为
```
python3 -m go2_sac.train --terrain flat --action-scale 0.6 \
    --steps 200_000 --checkpoint-freq 50_000     # 先学会在平地小跑
```
然后跑地形课程（从上面那个平地模型出发，逐级放开地形）：
```
python3 -m go2_sac.train --terrain steps --terrain-scale 0.70 --action-scale 0.6 \
    --learning-rate 2e-4 --fresh-reward --steps 250_000 \
    --save-dir models/go2_sac_steps_s0.70 --resume models/go2_sac_flat_xx/...
python3 -m go2_sac.train --terrain steps --terrain-scale 0.86 --action-scale 0.6 \
    --learning-rate 2e-4 --fresh-reward --reset-x -0.3 0.95 --steps 250_000 \
    --save-dir models/go2_sac_steps_s0.86 --resume models/go2_sac_steps_s0.70/model.zip
python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --action-scale 0.6 \
    --learning-rate 2e-4 --fresh-reward --reset-x -0.3 0.95 --steps 250_000 \
    --save-dir models/go2_sac_steps_s1.00 --resume models/go2_sac_steps_s0.86/model.zip
```

```
python3 -m go2_sac.train --terrain full --terrain-scale 0.56 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --reset-x 1.20 2.05 --steps 200_000 --checkpoint-freq 50_000 \
    --save-dir models/go2_sac_full_s0.56 --resume models/go2_sac_steps_s1.00/model.zip  # 首级 0.095 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.62 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --reset-x 1.20 2.05 --steps 250_000 --checkpoint-freq 50_000 \
    --save-dir models/go2_sac_full_s0.62 --resume models/go2_sac_full_s0.56/model.zip   # 0.105 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.70 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --reset-x 1.20 2.05 --steps 250_000 --checkpoint-freq 50_000 \
    --save-dir models/go2_sac_full_s0.70 --resume models/go2_sac_full_s0.62/model.zip   # 0.119 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.78 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --reset-x 1.20 2.05 --steps 250_000 --checkpoint-freq 50_000 \
    --save-dir models/go2_sac_full_s0.78 --resume models/go2_sac_full_s0.70/model.zip   # 0.133 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.86 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --reset-x 1.20 2.05 --steps 250_000 --checkpoint-freq 50_000 \
    --save-dir models/go2_sac_full_s0.86 --resume models/go2_sac_full_s0.78/model.zip   # 0.146 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.93 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --reset-x 1.20 2.05 --steps 300_000 --checkpoint-freq 50_000 \
    --save-dir models/go2_sac_full_s0.93 --resume models/go2_sac_full_s0.86/model.zip   # 0.158 m
python3 -m go2_sac.train --terrain full --terrain-scale 1.00 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --reset-x 1.20 2.05 --steps 350_000 --checkpoint-freq 50_000 \
    --save-dir models/go2_sac_full_s1.00 --resume models/go2_sac_full_s0.93/model.zip   # 原尺寸 0.170 m
```

#### 训练过程查看
```
tensorboard --logdir /path/to/unitree/runs
```

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

## GO2强化学习过地形（PPO）
和上面那节的**唯一区别是算法**：环境、地形、奖励、观测、课程分级、命令行参数、`config.json`
格式全都一样，`go2_common/` 里那份代码两边共用，只是模块名从 `go2_sac` 换成 `go2_ppo`。

命令行参数直接看上一节的 [#### 参数](#参数)：共用那张表一字不差，把「SAC 专属」那三个
（`--target-entropy` / `--gradient-steps` / `--seed-trot`）换成「PPO 专属」那五个
（`--n-steps` / `--batch-size` / `--n-epochs` / `--ent-coef` / `--target-kl`）即可。
训练、回放、奖励设置、课程分级各节的命令，把 `go2_sac` 改成 `go2_ppo` 同样成立。
