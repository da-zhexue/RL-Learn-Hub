python3 -m go2_sac.train --terrain flat --action-scale 0.6 --steps 300_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_flat    # 平地小跑

# 过槛课程训练
python3 -m go2_sac.train --terrain steps --terrain-scale 0.70 --action-scale 0.6 --learning-rate 2e-4 --fresh-reward --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_steps_s0.70 --resume models/go2_sac_flat/best/best_model
python3 -m go2_sac.train --terrain steps --terrain-scale 0.86 --action-scale 0.6 --learning-rate 2e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_steps_s0.86 --resume models/go2_sac_steps_s0.70/best/best_model
python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --action-scale 0.6 --learning-rate 2e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_steps_s1.00 --resume models/go2_sac_steps_s0.86/best/best_model

# 全地形课程训练
python3 -m go2_sac.train --terrain full --terrain-scale 0.56 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 200_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.56 --resume models/go2_sac_steps_s1.00/best/best_model  # 首级 0.095 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.62 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.62 --resume models/go2_sac_full_s0.56/best/best_model   # 0.105 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.70 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.70 --resume models/go2_sac_full_s0.62/best/best_model   # 0.119 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.78 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.78 --resume models/go2_sac_full_s0.70/best/best_model   # 0.133 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.86 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 250_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.86 --resume models/go2_sac_full_s0.78/best/best_model   # 0.146 m
python3 -m go2_sac.train --terrain full --terrain-scale 0.93 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 300_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s0.93 --resume models/go2_sac_full_s0.86/best/best_model   # 0.158 m
python3 -m go2_sac.train --terrain full --terrain-scale 1.00 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --reset-x -0.3 0.75 --steps 350_000 --checkpoint-freq 50_000 --save-dir models/go2_sac_full_s1.00 --resume models/go2_sac_full_s0.93/best/best_model   # 原尺寸 0.170 m

python3 -m go2_sac.train --terrain full --terrain-scale 1.00 --action-scale 0.7 --learning-rate 1e-4 --fresh-reward --steps 350_000 --checkpoint-freq 50_000 --resume models/go2_sac_full_s1.00/best/best_model   # 原尺寸 0.170 m