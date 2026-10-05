"""奖励函数：每一项一个独立函数，权重在 config.RewardCfg 里同名对应。

约定：
  * 每个 `_xxx(r, cfg)` 返回**该项的原始值**，不带权重；
  * 权重是 `RewardCfg` 里的同名字段（`reward.progress` <-> `RewardCfg.progress`）；
  * `compute_reward()` 负责加权求和，并返回每一项的分项值。

分项值会一路传进 `info` 再记到 TensorBoard。**不要省掉这一步** —— 只有总分曲线的时候，
你分不清"策略学会了站着不动"和"某一项写反了号"。

这里的项是按"能不能被刷分"筛过一轮的，加新项前先想清楚会不会被策略钻空子：
  * `climb` 用「地形高度增量」而不是「机身高度增量」，因为后者抬屁股／弹跳就能刷；
  * `progress` 在 |y|>1.25 和 x>3.5 处归零，否则绕到地形旁边走平地就能无限白拿；
  * 没有 `feet_air_time`：不加速度指令闸门时，它是"原地踏步"和"冲下悬崖"的经典刷分点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RewardCfg, quat_to_rpy


@dataclass
class RewardInput:
    """算一步奖励需要的全部量，由 env 填好。"""

    action: np.ndarray  # 本步动作（12 维）
    prev_action: np.ndarray  # 上一步动作
    q: np.ndarray  # 关节角，执行器顺序
    dq: np.ndarray  # 关节角速度
    tau: np.ndarray  # 关节力矩
    quat: np.ndarray  # 姿态四元数 w x y z
    base_v: np.ndarray  # 世界系线速度 (vx, vy, vz)
    base_w: np.ndarray  # 机体系角速度
    base_pos: np.ndarray  # 世界系位置 (x, y, z)
    prev_x: float  # 上一步的 x（progress 要差分）
    terrain_h: float  # 当前脚下的地形高度
    prev_terrain_h: float  # 上一步的地形高度
    terrain_level: int  # 当前站上第几级台阶（0=还在平地/槛上）
    prev_terrain_level: int  # 上一步的第几级
    cmd_vx: float  # 本回合的前进速度指令
    contact_force: float  # 非脚部件与地形的接触法向力之和
    terminated: bool  # 本步是否因为摔倒/成功而终止
    success: bool  # 本步是否到达终点


# ------------------------------------------------------------------ 前进 / 高度


def _corridor(y: float, cfg: RewardCfg) -> float:
    """走廊权重 w(y)：y=0 处为 1，到 |y|=progress_y_gate 平滑降到 0。

    **必须连续**。第一版写的是硬闸门（|y|>gate 时 Φ 直接置 0），势能塑形整体上确实
    telescoping、刷不了分，但单步上是个 ±300 的悬崖（x=3.0 处横跨一次闸门就是
    100×3.0 分），对 Q 来说是个极脏的目标。实测旧的 A=0.7 策略（会左右漂）在
    scale=0.56 上跑出来 progress = −376 / −664 / −654，就是这个悬崖刷出来的，
    根本不是"它在倒退 6 米"（场地一共才 4 米）。
    """
    a = min(abs(y) / max(cfg.progress_y_gate, 1e-6), 1.0)
    return 1.0 - a**3


def _potential_x(x: float, cfg: RewardCfg) -> float:
    """前进势能 Φ(s) = x（只算目标点之前的部分）。走廊权重**不在这里**乘，见 _progress。"""
    return min(x, cfg.progress_x_cap)


def _progress(r: RewardInput, cfg: RewardCfg) -> float:
    """前进奖励：**带折扣的 x 势能** γ·x' − x，而不是每步都付的速度年金。

    原来写的是 clip(vx/cmd, -1, 1)，vx=0 处为 0，看着很干净——但它**每一步都付**。
    实测（scale=0.56、5 个回合、A=0.7 的 best 模型）：

        回合 4（成功爬上 6 级）  224 步  总回报 +321   progress +258.2   climb +5.1
        回合 0（前进后摔倒）      174 步  总回报 +252   progress +273.8   climb -4.7
        回合 3（原地站满 1000 步） 1000 步  总回报  -44   progress  +11.0

    也就是说**爬完整段楼梯只值 55 分**（climb +5.1、level_bonus +30、success +20），
    而"在平地上快走一段"就值 250+。爬楼的收益比不爬只高 23%，而回合间方差有 370 分
    ——梯度当然说"别爬"。climb 权重 20.0 看着比 progress 的 2.0 大十倍，但 climb 只乘
    **每一步的高度增量**（0.095 m 量级），progress 却是每步都付、付两百多步。
    这是"卡在槛前不动反而分高"那个老毛病（见 _track_lin_vel）的同一个病根，
    当时只修了 vx=0 的地板，没修年金结构。

    改成势能后：站着不动收入**恒为 0**，"往前走"整段行程的总额有界（≈ x_cap - x_0），
    而**过了 x=2.1 想再拿一分就必须爬楼梯**——楼梯从"可选的加分项"变成"唯一收入来源"。

    **这里刻意不乘 γ**，和 _climb 不一样。势能塑形的教科书形式是 γΦ'−Φ，那是为了让
    折扣后的总额精确 telescope、不改变最优策略；但当 Φ 是**位置**这种沿路一直增长的量时，
    折扣会变成一笔按"停留时长"计的税：实测 50 Hz 下 Φ≈1.3，狗在**往前走**
    （Δx=0.0008 步）时单步仍是 0.995×1.2717−1.2725 = **−0.0071**，因为泄漏项
    (1−γ)Φ = 0.0064 比它大一个数量级。要打平得每步走 0.005·Φ 米，Φ=2.0 时就是 0.5 m/s、
    Φ=3.5 时要 0.875 m/s —— 比指令速度上限还高，整段行程净为负，站着 1000 步白扣 650 分。
    直接用 ΔΦ 就没有这个问题：展开后精确 telescope，站着不动恒为 0，来回走净值也是 0，
    刷不了分；代价是失去"折扣下的策略不变性"，而它在 Φ 增长时本来就已失效。

    走廊权重 w(y) 乘在**增量**上，不乘在累积的势能上。乘在势能上的话，一个跑到 x=3.83、
    但横向漂到 |y|=1.42 的回合会在**结算时**把前面攒的势能全部吐回去（w 归零），
    实测那一回合 progress = -174 —— 明明净前进了 2.2 m。平滑化只抹掉单步跳变，
    抹不掉"结算没收"。乘法改成对增量后：走廊外 Δx 挣 0 分（照样防住绕旁边平地白拿），
    但**漂出去不会倒扣已经挣到的**。它在走廊内仍然是 telescoping 的，来回走净值 0。
    """
    prev = _potential_x(float(r.prev_x), cfg)
    cur = _potential_x(float(r.base_pos[0]), cfg)
    return float((cur - prev) * _corridor(float(r.base_pos[1]), cfg))


def _track_lin_vel(r: RewardInput, cfg: RewardCfg) -> float:
    """速度跟踪，高斯型，最优点就是指令速度。**尾部要减掉 vx=0 处的取值**。

    不减的话，vx=0（原地不动）也能拿到 exp(-cmd²/σ²)：指令 0.4 时是 0.19/步，
    1000 步的回合白拿 190 分。实测零动作站着不动总回报 +148.7，
    比"走到槛前卡住"（-1.6）还高，"爬上去卡住"更是 -125 —— 三个状态的排序完全反了，
    策略停在槛前不动**是它算出来的最优解**。这正是最初那个
    「0.92~0.95 时狗卡在槛下后不动了，但奖励比迈过槛还高」的成因。

    `track_sigma` 从 0.25 收到 0.1 只把 c̲m̲d̲=0.8 时的地板压到 0.002，
    cmd=0.4 时仍有 0.19 —— 因为指令是 (0.4, 0.8) 均匀采样的，光收 σ 治不了低指令那一半。
    减掉地板后 vx=0 精确为 0，与 σ 和 cmd 都无关。
    """
    floor = float(np.exp(-(r.cmd_vx**2) / cfg.track_sigma))
    return float(max(0.0, np.exp(-((r.base_v[0] - r.cmd_vx) ** 2) / cfg.track_sigma) - floor))


def _climb(r: RewardInput, cfg: RewardCfg) -> float:
    """爬升奖励：**带折扣的地形势能** γ·h(x') − h(x)。

    写 Δh 而不用 Δbase_z，是为了防止抬屁股/弹跳刷分（机身高度随便就能抬，地形高度不行）。

    权重必须给大：0.08 m 的槛乘 2.0 只有 0.16 分，而卡在槛上时 height/collision 惩罚
    一步就 -0.13，爬上去反而是亏的 —— 实测平地策略在槛上卡住 -125 分、在槛前卡住只 -1.6 分，
    **策略选择不爬完全正确**。

    `climb_gamma` 现在默认 1.0（即不乘 γ），原因和 _progress 那段写的一样：h 最高 0.92 m，
    看着泄漏不大，但权重提到 100 之后 (1−γ)·h·100 = 0.46 分/步，站在台阶上 1000 步就是 -460。
    """
    return float(cfg.climb_gamma * r.terrain_h - r.prev_terrain_h)


def _level_bonus(r: RewardInput, cfg: RewardCfg) -> float:
    """每登上一级新台阶给一次性的分。

    climb 是势能增量，只在"跨上台阶沿的那一步"出现一次，很容易被动作噪声淹没；
    台阶数是个**离散**信号，直接告诉策略"你上了一个新高度"，比连续势能好学好认。
    只涨不跌：掉下台阶不重复扣（掉下去由 climb 项罚），否则一次失误会被连罚两遍。
    """
    return float(max(0, r.terrain_level - r.prev_terrain_level))


def _base_height(r: RewardInput, cfg: RewardCfg) -> float:
    """机身离当地地形的高度**偏差**（正值，权重是负的 -> 惩罚）。平地和顶平台目标值相同。

    这里故意返回线性偏差，而不是原来的 exp(-err²)：
      * exp 形式在站姿正确时给满额正分，等于给"站着不动"发保底工资。实测零动作站立能白拿
        +0.50/步（占总分 74%），策略几步就能找到这个盆地，之后再也不肯冒险尝试走路
        ——1e6 步训完平均 vx 只有指令的 1/10，脚最多抬 7 cm，上不了 8 cm 的槛。
        偏差形式站对了正好是 0，不给保底，走路（progress+track 约 3 分/步）才是唯一的大额收入。
      * exp 一旦偏出 0.2 m 就几乎没有梯度，惩罚不了"趴地上往前爬"这类病态姿态；
        线性偏差始终有梯度。
    参考实现（legged_gym / Isaac Lab）的 base_height 也是惩罚而不是奖励。

    **死区**（height_dead_zone）：跨台阶时狗是"前脚在上、后脚在下"的跨坐姿势，
    机身必然比目标低一截，这是正确动作而不是错误姿态。没有死区的话，爬上去反而一直在挨罚，
    实测"卡在槛上"整回合 -125 分，比"卡在槛前不动"的 -1.6 分差得多，策略于是学会不爬。
    死区内的偏差不计分，让爬台阶的过程不再挨罚。
    """
    err = abs(float(r.base_pos[2]) - r.terrain_h - cfg.height_target)
    return float(max(0.0, err - cfg.height_dead_zone))


def _lateral(r: RewardInput, cfg: RewardCfg) -> float:
    """别跑偏到地形外面去。"""
    return float(r.base_pos[1] ** 2)


# ------------------------------------------------------------------ 姿态


def _orientation(r: RewardInput, cfg: RewardCfg) -> float:
    """机身保持水平。"""
    roll, pitch, _ = quat_to_rpy(r.quat)
    return float((1.0 - np.cos(roll)) + (1.0 - np.cos(pitch)))


def _yaw(r: RewardInput, cfg: RewardCfg) -> float:
    """航向对准 +x。地形关于 y 对称、外面又是无限平地，不压住航向就会横着漂出去。"""
    yaw = quat_to_rpy(r.quat)[2]
    return float(1.0 - np.cos(yaw))


def _yaw_rate(r: RewardInput, cfg: RewardCfg) -> float:
    """别原地打转。"""
    return float(r.base_w[2] ** 2)


# ------------------------------------------------------------------ 平滑 / 能耗


def _action_rate(r: RewardInput, cfg: RewardCfg) -> float:
    """动作别抖。"""
    return float(np.sum((r.action - r.prev_action) ** 2))


def _torques(r: RewardInput, cfg: RewardCfg) -> float:
    """省点力矩。"""
    return float(np.sum(r.tau**2))


def _collision(r: RewardInput, cfg: RewardCfg) -> float:
    """非脚部件（大腿、小腿、机身）撞地形的惩罚，按接触力缩放。

    注意这里只是惩罚、**不终止**：Go2 前部的载荷几何体比脚底还低约 1 cm，
    爬第一级台阶时擦一下是正常的，罚狠了会把"抬腿够台阶"这个正确动作一起罚掉。

    **要减掉一个底噪**：狗用小腿顶着台阶沿往上蹭的时候，接触力稳定在几十牛，
    那是它在爬，不是它在撞。原样计费的话"卡住"这个状态每步要赔 0.13 分，
    整回合累到 -125 —— 而卡在台阶前面不动是 0 分。策略于是学会**根本不靠近台阶**，
    这正是"走到 x≈0.94 就停住、20 万步毫无进展"的直接原因。
    减掉阈值后，蹭台阶是免费的，只有真正的撞击才罚。
    """
    return float(max(0.0, r.contact_force - cfg.collision_force_free))


# ------------------------------------------------------------------ 终止


def _success(r: RewardInput, cfg: RewardCfg) -> float:
    return 1.0 if r.success else 0.0


def _fall(r: RewardInput, cfg: RewardCfg) -> float:
    """摔倒/出界终止（成功不算）。"""
    return 1.0 if (r.terminated and not r.success) else 0.0


# 顺序即打印顺序：正向项在前，惩罚项在后，终止项垫底。
TERMS = (
    ("progress", _progress),
    ("track_lin_vel", _track_lin_vel),
    ("climb", _climb),
    ("level_bonus", _level_bonus),
    ("base_height", _base_height),
    ("lateral", _lateral),
    ("orientation", _orientation),
    ("yaw", _yaw),
    ("yaw_rate", _yaw_rate),
    ("action_rate", _action_rate),
    ("torques", _torques),
    ("collision", _collision),
    ("success", _success),
    ("fall", _fall),
)


def compute_reward(r: RewardInput, cfg: RewardCfg) -> tuple[float, dict[str, float]]:
    """加权求和，返回 (总奖励, {项名: 加权后的分项值})。"""
    parts = {name: float(getattr(cfg, name)) * float(fn(r, cfg)) for name, fn in TERMS}
    return float(sum(parts.values())), parts
