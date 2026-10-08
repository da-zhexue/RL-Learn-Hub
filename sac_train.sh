# python3 -m go2_sac.train --terrain flat --action-scale 0.6 --steps 100_000 --checkpoint-freq 20_000 --save-dir models/go2_sac_flat    # 平地小跑

# 过槛课程训练
python3 -m go2_sac.train --terrain steps --terrain-scale 0.70 --action-scale 0.6 --learning-rate 2e-4 --fresh-reward --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_steps_s0.70 --resume models/go2_sac_flat_20261007_160920/checkpoints/rl_50000_steps.zip
python3 -m go2_sac.train --terrain steps --terrain-scale 0.86 --action-scale 0.6 --learning-rate 2e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_steps_s0.86 --resume models/go2_sac_steps_s0.70/model.zip
python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --action-scale 0.6 --learning-rate 2e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_steps_s1.00 --resume models/go2_sac_steps_s0.86/model.zip

# 全地形课程训练
# --reset-x 一律避开地形棱：出生点按**整只狗的脚印**取最大地形高度（env.reset -> top_footprint，
# 半长 0.24），生在台阶棱上 = 悬空半只脚落地 -> 立刻侧翻。旧值 1.20~2.05 整个区间都压在
# 第一个槛(1.1~1.3)/第一级台阶(2.1)上，10 个评估回合有 7 个在 25~59 步内 roll≈-1.0 结束（AGENT.md §5-C1）。
# 0.75 + 0.24 = 0.99 < 1.1，整只狗落在平地上。
python3 -m go2_sac.train --terrain full --terrain-scale 0.56 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 200_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.56 --resume models/go2_sac_steps_s1.00/model.zip  # 首级 0.095 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.62 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.62 --resume models/go2_sac_full_s0.56/model.zip   # 0.105 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.70 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.70 --resume models/go2_sac_full_s0.62/model.zip   # 0.119 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.78 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.78 --resume models/go2_sac_full_s0.70/model.zip   # 0.133 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.86 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.86 --resume models/go2_sac_full_s0.78/model.zip   # 0.146 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.93 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 300_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.93 --resume models/go2_sac_full_s0.86/model.zip   # 0.158 m
python3 -m go2_sac.train --terrain full --terrain-scale 1.00 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 350_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s1.00 --resume models/go2_sac_full_s0.93/model.zip   # 原尺寸 0.170 m

python3 -m go2_sac.train --terrain full --terrain-scale 1.00 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --steps 350_000 --checkpoint-freq 50_000 --resume models/go2_sac_full_s1.00/model.zip   # 原尺寸 0.170 m