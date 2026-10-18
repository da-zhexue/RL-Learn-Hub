# =====================================================================
# Go2 PPO 训练流水线（PowerShell 语法；在本机用 pwsh -File ppo_train.sh 或直接粘到 PowerShell）
# 文件名叫 .sh 是历史遗留，内容一直是 PowerShell 续行（反引号）。
#
# 目的：先修步态（抬脚 + 站高 + 四条腿都迈），再上 0.08 m 槛，最后爬台阶。
# 每阶段跑完先验收，不达标就留在本阶段继续加训，别往下走：
#   & $PY -m go2_ppo.diag policy models/<本阶段>/best/best_model.zip <地形> 1.0
# 平地验收线：四脚 clearance 中位 >= 0.03、最高 >= 0.08；机身 >= 0.26 m；
#             肚子最低点 >= 0.15 m；速度比 0.9~1.1；0 摔倒。
# 台阶验收线：best_model 20 回合成功率 >= 60%，再把 terrain-scale 抬 0.2。
#
# 注意（AGENT.md 5-A2/A3/D1/D2）：改奖励/换阶段必须 --fresh-reward；--steps 是本阶段
# 追加步数不是累计；看 best/best_model.zip 不要直接拿 model.zip 进下一阶段；
# 同一时间只跑一个训练（两个 8 环境任务会把 16 核拖慢一倍）。
# =====================================================================

$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location $PSScriptRoot
$PY = 'D:\anaconda\envs\pytorch\python.exe'

# ---------------------------------------------------------------------
# 阶段 1：平地修步态（clearance 0.08 / 机身 0.29 / 四条腿都要摆动）
#   起点：gait2 最终模型（会走路但 RR 脚拖地，见 AGENT.md 5-B9）
#   新项：feet_air_time +6.0、foot_swing_height +10.0（瓶颈式，最差那只脚决定分数）
# ---------------------------------------------------------------------
& $PY -m go2_ppo.train `
    --resume models/go2_ppo_flat_gait2/model.zip `
    --fresh-reward --terrain flat --action-scale 0.7 --height-scan `
    --steps 2000000 --checkpoint-freq 250000 --eval-freq 50000 `
    --ent-coef 0.005 `
    --save-dir models/go2_ppo_flat_gait3


# ---------------------------------------------------------------------
# 阶段 1b：平地继续（gait3 两百万步后：机身 0.197->0.240、RR 抬脚 0.002->0.038，
#   但后腿抬得还不够高 —— 四脚要 >=0.08 才达标。同一套权重继续，让瓶颈项把后腿推上去）
#   注：gait4 在 13.7M（+500k）处被关机器中断，这里从最新 checkpoint 续跑剩下的 1.5M。
#   全新的机器上没有 gait4 checkpoint 时，改成 --resume models/go2_ppo_flat_gait3/best/best_model.zip
#   并把 --steps 设成 2000000。
# ---------------------------------------------------------------------
& $PY -m go2_ppo.train `
    --resume models/go2_ppo_flat_gait4/checkpoints/rl_13701488_steps.zip `
    --fresh-reward --terrain flat --action-scale 0.7 --height-scan `
    --steps 1500000 --checkpoint-freq 250000 --eval-freq 50000 `
    --ent-coef 0.003 `
    --save-dir models/go2_ppo_flat_gait4b

# ---------------------------------------------------------------------
# 阶段 2：0.08 m 槛（steps 地形，clearance_target 自动取 0.08）
# ---------------------------------------------------------------------
& $PY -m go2_ppo.train `
    --resume models/go2_ppo_flat_gait4b/best/best_model.zip `
    --fresh-reward --terrain steps --terrain-scale 1.0 --action-scale 0.7 --height-scan `
    --steps 1500000 --checkpoint-freq 250000 --eval-freq 50000 `
    --ent-coef 0.003 `
    --save-dir models/go2_ppo_steps1

# ---------------------------------------------------------------------
# 阶段 3：台阶课程 0.4 -> 0.6 -> 0.8 -> 1.0，每档 1.5M~2M 步
# ---------------------------------------------------------------------
& $PY -m go2_ppo.train `
    --resume models/go2_ppo_steps1/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 0.4 --action-scale 0.7 --height-scan `
    --steps 1500000 --checkpoint-freq 250000 --eval-freq 50000 `
    --save-dir models/go2_ppo_full_s04

& $PY -m go2_ppo.train `
    --resume models/go2_ppo_full_s04/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 0.6 --action-scale 0.7 --height-scan `
    --steps 1500000 --checkpoint-freq 250000 --eval-freq 50000 `
    --save-dir models/go2_ppo_full_s06

& $PY -m go2_ppo.train `
    --resume models/go2_ppo_full_s06/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 0.8 --action-scale 0.7 --height-scan `
    --steps 1500000 --checkpoint-freq 250000 --eval-freq 50000 `
    --save-dir models/go2_ppo_full_s08

& $PY -m go2_ppo.train `
    --resume models/go2_ppo_full_s08/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 1.0 --action-scale 0.7 --height-scan `
    --steps 2000000 --checkpoint-freq 250000 --eval-freq 50000 `
    --save-dir models/go2_ppo_full_s10
