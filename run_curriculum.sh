#!/usr/bin/env bash
# 地形课程：从一个已经会走路的策略出发，把地形高度从矮到高逐级放开。
#
# 为什么要逐级而不是直接上原尺寸：直接跳过去策略只会停在槛前吃保底分，
# 而"停在槛前"这件事在旧奖励下确实是它的最优解（见 README 的奖励一节）。
# 现在奖励修好了，但地形难度仍然要一级一级加——0.08 m 的槛要求后腿摆幅比
# 0.056 m 的槛再大一截，平地策略从没见过这种要求。
#
# 用法：./run_curriculum.sh models/go2_sac_flat_xxx/checkpoints/rl_800000_steps.zip
#
# 已经练完的级会按输出目录自动跳过，所以中途被打断后原样重跑即可续上。
set -euo pipefail
cd "$(dirname "$0")"

# action_scale 取 0.6 不是 0.8。抬脚表（diag lift）说 0.8 的后脚在 0.7 幅就有 0.10~0.11、
# 比 0.08 的槛宽裕得多，看起来该用 0.8；但整条课程实测下来 0.8 每一级都更差：
#     A=0.6: scale 0.5 ✓(0.85~1.33×)  0.7 ✓(0.86~1.07×)  1.0 ✗
#     A=0.8: scale 0.5 ✓(1.03×)       0.7 ~(0.67~1.10×)  0.8 起就在 ±200 之间震荡
# 多出来的摆幅买到了抬脚余量，却把可控性赔了进去（同样的动作值对应更大的关节位移，
# 姿态更容易被噪声推出稳定域），策略把容量全花在保持平衡上，反而没余力去爬。
A=0.6
LR=2e-4          # 比从头训练的 3e-4 小一点，避免把已经会走路的策略打散
INPUT=${1:?用法: run_curriculum.sh <起始模型 model.zip>}

# 找一个运行目录里当前可用的模型：优先最终的 model.zip，其次 best，最后最新的 checkpoint
model_of() {
  local d=$1
  if [ -f "$d/model.zip" ]; then echo "$d/model.zip"
  elif [ -f "$d/best/best_model.zip" ]; then echo "$d/best/best_model.zip"
  else ls -1 "$d"/checkpoints/*.zip 2>/dev/null | sort -V | tail -1
  fi
}

n_of() { python3 -c "from stable_baselines3 import SAC; print(SAC.load('$1', device='cpu').num_timesteps)"; }

mkdir -p logs
echo "===== 课程开始 $(date '+%F %T')  起始模型 $INPUT  已有 $(n_of "$INPUT") 步" | tee -a logs/curriculum.log

# 注意 --steps 的语义：续训时 SB3 执行 `total_timesteps += self.num_timesteps`
# （base_class.py:416），所以这里是**本阶段再练多少步**，不是累计目标。
# 按累计算的话每一级都会被放大成几百万步，课程根本推不到最后一级
# （第一次跑课程就是这么废掉的：想练 25 万步，实际练了 100 万）。
#
# --reset-x 可选：默认 (-0.3, 0.3)，出生点全在平地起点。最难的那几级放宽到 0.95，
# 让一部分回合直接生在槛前 15 cm 处——不然一个 20 秒的回合大部分时间都在平地上走，
# 真正"撞槛-抬腿-上槛"的尝试只有最后两三秒，样本效率极低。
# 不直接把出生点全挪到槛前，是要保留平地行走的练习，否则会把已经学会的步态忘掉。
run() {  # $1=地形 $2=scale $3=本阶段步数 $4=输出目录 [$5=reset_x_hi]
  local terrain=$1 scale=$2 inc=$3 out=$4 rx=${5:-}
  if [ -n "$(model_of "$out")" ]; then
    INPUT=$(model_of "$out")
    echo "----- $terrain scale=$scale 已存在，跳过（用 $INPUT）" | tee -a logs/curriculum.log
    return
  fi
  local extra=()
  [ -n "$rx" ] && extra=(--reset-x -0.3 "$rx")
  echo "----- $terrain scale=$scale  本阶段 $inc 步  reset_x=-0.3~${rx:-0.3}  $(date '+%F %T')" \
      | tee -a logs/curriculum.log
  # --fresh-reward 必须加：否则 checkpoint 里旧的奖励权重（climb=2.0、没有高度死区、
  # track 还带着"站着不动"的地板）会被静默恢复回来，代码改了也等于没改。
  python3 -u -m go2_sac.train \
      --terrain "$terrain" --terrain-scale "$scale" --action-scale "$A" \
      --fresh-reward --learning-rate "$LR" --steps "$inc" \
      "${extra[@]}" --save-dir "$out" --resume "$INPUT" 2>&1 | tee -a logs/curriculum.log
  INPUT=$(model_of "$out")
}

# 先只放两个槛（steps 地形没有台阶），把"抬腿过槛"这件事单独学会。
# 0.70 → 1.00 拆成四级：这一段要求后腿摆幅从"够用"跳到"接近饱和"，
# 一次跳 0.7→1.0 会直接崩（实测回报 15 → -194、速度比掉到 0.01×、卡死在第一道槛）。
run steps 0.70 250000 models/cur_s070_steps                  # 0.056 m，已练好
run steps 0.78 250000 models/cur3_s078_steps                 # 0.062 m
run steps 0.86 250000 models/cur3_s086_steps 0.95            # 0.069 m
run steps 0.93 300000 models/cur3_s093_steps 0.95            # 0.074 m
run steps 1.00 400000 models/cur3_s100_steps 0.95            # 0.080 m —— 关键一级
# 再把六级台阶放回来，同样从矮到高
run full  0.60 300000 models/cur3_s060_full 0.95
run full  0.80 300000 models/cur3_s080_full 0.95
run full  1.00 400000 models/cur3_s100_full                  # 原尺寸：槛 0.08、台阶每级 0.15

echo "===== 课程结束 $(date '+%F %T')  最终模型 $(model_of models/cur3_s100_full)" \
    | tee -a logs/curriculum.log
