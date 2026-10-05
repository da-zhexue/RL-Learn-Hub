#!/usr/bin/env bash
# 楼梯阶段：从"已经会过两道 0.08 m 槛"的策略出发，专攻 x≥2.1 的六级台阶。
#
# 用法：./run_stairs.sh models/st_s056_full/best/best_model.zip
#
# ── 为什么要单独一个脚本、把出生点挪到楼梯口 ──────────────────────────────
# `full` 地形的走向是 平地(-0.3~1.1) -> 两道 0.08 槛(1.1~1.7) -> 六级台阶(2.1~3.66)。
# 出生点留在起点的话，一个 20 秒回合里前面十几秒都在重复已经会的东西，
# 真正碰到台阶只剩最后两三秒。把出生点放到楼梯前，同一档位的评估立刻从
# "速度比 0.04×"回到正常行走。但**必须留出助跑距离**：只留 5 cm 的话，
# 狗从 0.30 m 高处落地、还在落稳时就已经顶在台阶面上，既没速度也没摆幅。
#
# ── 台阶几何（scene.xml，scale=1）────────────────────────────────────────
# 顶面 0.17/0.32/0.47/0.62/0.77/0.92 —— **第一级就抬 0.17 m**，之后每级 0.15 m，
# 而踏面只有约 0.19 m（方块互相重叠）。这是很陡的楼梯，也是全程最难的一段。
#
# ── action_scale 为什么是 0.7 ────────────────────────────────────────────
# 拿 0.35 档（A=0.6，5/5 成功率）训出来的模型，换不同 action_scale 去跑 scale=0.50
# （首级台阶 0.085 m），**不做任何训练**直接看第一次评估：
#     A=0.6: 0/5  速度比 0.02×  最高台阶 4/6      <- A=0.6 在楼梯上的天花板只有 ~0.07 m
#     A=0.7: 0/5  速度比 0.53×  最高台阶 6/6      <- 没训过就上到了顶
#     A=0.8: 最高台阶 2/6                          <- 开始震荡
#     A=1.0: 最高台阶 1/6
# A=0.6 天花板低是因为台阶要求"每一级都把后脚抬到 0.085"，而 0.7 动作幅度下后脚底
# 只有 0.065：单次靠贴饱和能蒙对（所以能过孤立的 0.08 槛），连着六次不行。
set -euo pipefail
cd "$(dirname "$0")"

A=0.7
# 微调用 1e-4。**这个值以前根本传不进去**：SB3 的 SAC.load 会先用 checkpoint 里存的
# 超参覆盖 model.__dict__，而 SAC.load() 没传 kwargs，于是命令行给的 learning_rate
# 被静默忽略——所有模型存下来的 lr 一直是建模型时的 3e-4。实测（都在 scale 0.62、
# 从同一个模型续训 5 万步）：3e-4 -> 回报 62.8 / 台阶 1/6；1e-4 -> 122.5 / 6/6；
# 3e-5 -> 84.0 / 4/6。3e-4（一直实际在用的那个）恰恰最差，这就是"训练反而把模型
# 练坏"的主因。修法见 train.py 里续训那段的注释（光赋值 learning_rate 还不够，
# 要连 lr_schedule 一起换，否则三个优化器一个都不动）。
LR=1e-4
RX_LO=1.20       # 出生点：楼梯前，留出 0~0.9 m 助跑（x∈[1.7,2.1] 是仅有的一段平路）
RX_HI=2.05
INPUT=${1:?用法: run_stairs.sh <起始模型 model.zip>}

# 取这个阶段**最好**的模型，而不是最后一个。
# 两个在顺的阶段里差不多，但在难度顶到天花板的阶段差得很远——实测：
#     st_s056  model.zip 回报 111 / 成功率 1/5   vs  best_model.zip 回报 269 / 4/5
# 因为难度一超能力，训练就开始把策略往"别冒险"上拽（摔倒会终止回合、丢掉剩下九成
# 的回报），于是最后一个 checkpoint 恰恰是最差的。上一版脚本优先取 model.zip，
# 等于每级都拿退化模型去喂下一级，越滚越糟。
model_of() {
  local d=$1
  if [ -f "$d/best/best_model.zip" ]; then echo "$d/best/best_model.zip"
  elif [ -f "$d/model.zip" ]; then echo "$d/model.zip"
  else ls -1 "$d"/checkpoints/*.zip 2>/dev/null | sort -V | tail -1
  fi
}
n_of() { python3 -c "from stable_baselines3 import SAC; print(SAC.load('$1', device='cpu').num_timesteps)"; }

mkdir -p logs
echo "===== 楼梯课程开始 $(date '+%F %T')  起始模型 $INPUT  已有 $(n_of "$INPUT") 步" \
    | tee -a logs/curriculum.log

# 注意 --steps 的语义：续训时 SB3 执行 `total_timesteps += self.num_timesteps`
# （base_class.py），所以这里是**本阶段再练多少步**，不是累计目标。
run() {  # $1=scale $2=本阶段步数 $3=输出目录 $4=首级台阶高（仅用于打印）
  local scale=$1 inc=$2 out=$3
  if [ -n "$(model_of "$out")" ]; then
    INPUT=$(model_of "$out")
    echo "----- full scale=$scale 已存在，跳过（用 $INPUT）" | tee -a logs/curriculum.log
    return
  fi
  echo "----- full scale=$scale  首级台阶 $4 m  本阶段 $inc 步  $(date '+%F %T')" \
      | tee -a logs/curriculum.log
  python3 -u -m go2_sac.train \
      --terrain full --terrain-scale "$scale" --action-scale "$A" --fresh-reward \
      --learning-rate "$LR" --steps "$inc" --reset-x "$RX_LO" "$RX_HI" \
      --save-dir "$out" --resume "$INPUT" 2>&1 | tee -a logs/curriculum.log
  INPUT=$(model_of "$out")
}

# 输出目录带 st3_ 前缀：前两轮的 st_* / st2_* 结果原样保留，便于对比。
# 这一轮和前两轮的区别只有一处：**出生穿模的 bug 修好了**（env.py 的 reset 改成按
# 整只狗的脚印取最大地形高度）。之前每一个出生点都可能把脚生成在地形内部（实测
# 24 个点里 17 个有脚被埋，最深 9.15 cm），而穿深≈棱高，**地形越高越严重**——
# 也就是说整条楼梯课程一直是在"出生即穿模、被接触力弹翻"的数据上训练的。
#
# 阶梯每级只涨 1.08~1.14 倍——0.35 -> 0.50 -> 0.65 -> 0.80 -> 1.00（每级 1.4~1.25×）
# 那种粗阶梯实测第一跳就崩：0.059 -> 0.085 看着只多 2.6 cm，但 steps 上练的是
# **孤立**的槛（前后有平地可以重新站稳），这里是六个连着的。
run 0.56 200000 models/st3_s056_full 0.095   # 已有模型在此能拿 4/5
run 0.62 250000 models/st3_s062_full 0.105
run 0.70 250000 models/st3_s070_full 0.119
run 0.78 250000 models/st3_s078_full 0.133
run 0.86 250000 models/st3_s086_full 0.146
run 0.93 300000 models/st3_s093_full 0.158
run 1.00 350000 models/st3_s100_full 0.170   # 原尺寸：槛 0.08、台阶每级 0.15

echo "===== 楼梯课程结束 $(date '+%F %T')  最终模型 $(model_of models/st2_s100_full)" \
    | tee -a logs/curriculum.log
