python3 -m go2_sac.train --terrain flat --action-scale 0.6 --steps 300_000 \
    --checkpoint-freq 50_000 --save-dir models/go2_sac_flat

python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --action-scale 0.6 \
    --learning-rate 2e-4 --fresh-reward --steps 2_000_000 --checkpoint-freq 200_000 \
    --save-dir models/go2_sac_steps_a0.6 --resume models/go2_sac_flat/best/best_model

python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --action-scale 0.7 \
    --learning-rate 2e-4 --fresh-reward --steps 400_000 --checkpoint-freq 100_000 \
    --save-dir models/go2_sac_steps_a0.7 --resume models/go2_sac_steps_a0.6/best/best_model

python3 -m go2_sac.train --terrain full --terrain-scale 0.40 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --steps 2_000_000 --checkpoint-freq 500_000 \
    --reset-frac-stair 0.5 --reset-x-stair 1.40 1.85 --reset-replay-buffer \
    --save-dir models/go2_sac_full_0.4 --resume models/go2_sac_steps_a0.7/best/best_model 

python3 -m go2_sac.train --terrain full --terrain-scale 0.60 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --steps 500_000 --checkpoint-freq 100_000 \
    --reset-frac-stair 0.5 --reset-x-stair 1.40 1.85 \
    --save-dir models/go2_sac_full_0.6 --resume models/go2_sac_full_0.4/best/best_model 

python3 -m go2_sac.train --terrain full --terrain-scale 0.80 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --steps 500_000 --checkpoint-freq 100_000 \
    --reset-frac-stair 0.5 --reset-x-stair 1.40 1.85 \
    --save-dir models/go2_sac_full_0.8 --resume models/go2_sac_full_0.6/best/best_model 
