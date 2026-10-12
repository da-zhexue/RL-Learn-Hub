python3 -m go2_sac.train --terrain flat --action-scale 0.6 --steps 300_000 \
    --checkpoint-freq 50_000 --save-dir models/go2_sac_flat

python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --action-scale 0.6 \
    --learning-rate 2e-4 --fresh-reward --steps 2_000_000 --checkpoint-freq 200_000 \
    --save-dir models/go2_sac_steps_a0.6 --resume models/go2_sac_flat/best/best_model

python3 -m go2_sac.train --terrain steps --terrain-scale 1.00 --action-scale 0.7 \
    --learning-rate 2e-4 --fresh-reward --steps 300_000 --checkpoint-freq 50_000 \
    --save-dir models/go2_sac_steps_a0.7 --resume models/go2_sac_steps_a0.6/best/best_model

python3 -m go2_sac.train --terrain full --terrain-scale 1.00 --action-scale 0.7 \
    --learning-rate 1e-4 --fresh-reward --steps 2_000_000 --checkpoint-freq 200_000 \
    --save-dir models/go2_sac_full --resume models/go2_sac_steps_a0.7/best/best_model 
