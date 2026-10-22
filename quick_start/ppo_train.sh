# mujoco+gym的ppo训练迁移到Windows上了，所以下面的分行改为了 ` ，不能在linux上直接运行该sh。

# 训练阶段 1：平地
python -m go2_ppo.train `
    --fresh-reward --terrain flat --action-scale 0.7 --height-scan `
    --steps 20_000_000 --checkpoint-freq 2_500_000 --eval-freq 50_000 `
    --ent-coef 0.005 `
    --save-dir models/go2_ppo_flat

# 训练阶段 2：台阶
python -m go2_ppo.train `
    --resume models/go2_ppo_flat/best/best_model.zip `
    --fresh-reward --terrain steps --terrain-scale 1.0 --action-scale 0.7 --height-scan `
    --steps 1_500_000 --checkpoint-freq 250_000 --eval-freq 50_000 `
    --ent-coef 0.003 `
    --save-dir models/go2_ppo_steps1

# 训练阶段 3：全高
python -m go2_ppo.train `
    --resume models/go2_ppo_steps1/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 0.4 --action-scale 0.7 --height-scan `
    --steps 1_500_000 --checkpoint-freq 250_000 --eval-freq 50_000 `
    --save-dir models/go2_ppo_full_s04

python -m go2_ppo.train `
    --resume models/go2_ppo_full_s04/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 0.6 --action-scale 0.7 --height-scan `
    --steps 1_500_000 --checkpoint-freq 250_000 --eval-freq 50_000 `
    --save-dir models/go2_ppo_full_s06

python -m go2_ppo.train `
    --resume models/go2_ppo_full_s06/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 0.8 --action-scale 0.7 --height-scan `
    --steps 1_500_000 --checkpoint-freq 250_000 --eval-freq 50_000 `
    --save-dir models/go2_ppo_full_s08

python -m go2_ppo.train `
    --resume models/go2_ppo_full_s08/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 1.0 --action-scale 0.7 --height-scan `
    --steps 2_000_000 --checkpoint-freq 250_000 --eval-freq 50_000 `
    --save-dir models/go2_ppo_full_s10

# 阶段 4：修复
python -m go2_ppo.train `
    --resume models/go2_ppo_full_s10/best/best_model.zip `
    --fresh-reward --terrain full --terrain-scale 0.9 --action-scale 0.7 --height-scan `
    --steps 2_000_000 --checkpoint-freq 250_000 --eval-freq 50_000 `
    --ent-coef 0.003 `
    --save-dir models/go2_ppo_full_s09

python -m go2_ppo.train `
    --resume models/go2_ppo_full_s09/model.zip `
    --fresh-reward --terrain full --terrain-scale 1.0 --action-scale 0.7 --height-scan `
    --steps 2_000_000 --checkpoint-freq 250_000 --eval-freq 50_000 `
    --ent-coef 0.003 `
    --reward-feet-clearance 0.5 --reward-feet-air-time 2 --reward-foot-swing-height 2 `
    --reward-climb 800 --reward-level-bonus 60 `
    --reward-orientation -2.0 `
    --save-dir models/go2_ppo_full_s10b
