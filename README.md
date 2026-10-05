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
```
go2_sac/config.py    常量、EnvCfg/RewardCfg/TrainCfg 配置类、四元数工具
go2_sac/terrain.py   课程变体、地形高度查询、成功/摔倒判据
go2_sac/env.py       Go2TerrainEnv（Gymnasium 环境）
go2_sac/reward.py    奖励函数：每项一个函数，权重在 RewardCfg 里同名对应
go2_sac/train.py     SAC 训练入口
go2_sac/play.py      回放（viewer / dds）
go2_sac/diag.py      诊断工具（地形/站姿/抬脚/奖励分项/回合为什么结束）
```

### 训练
从头训：
```
python3 -m go2_sac.train --terrain flat --steps 600_000       # 先学会在平地小跑
```
然后跑地形课程（从上面那个平地模型出发，逐级放开地形）：
```
./run_curriculum.sh models/go2_sac_flat_xx/checkpoints/rl_600000_steps.zip
```
手工逐级等价于（`--steps` 是本级再练多少步）：
```
python3 -m go2_sac.train --terrain steps --terrain-scale 0.70 --fresh-reward \
    --steps 250_000 --resume models/go2_sac_flat_xx/...               # 5.6 cm 的槛
python3 -m go2_sac.train --terrain steps --terrain-scale 0.86 --fresh-reward \
    --reset-x -0.3 0.95 --steps 250_000 --resume models/cur_s070_steps/model.zip
python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --fresh-reward \
    --reset-x -0.3 0.95 --steps 400_000 --resume models/cur_s086_steps/model.zip  # 8 cm 的槛
python3 -m go2_sac.train --terrain full  --terrain-scale 1.00 --fresh-reward \
    --steps 400_000 --resume models/cur_s100_steps/model.zip          # 再加六级台阶
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
