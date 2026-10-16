python -m go2_ppo.train --terrain flat --action-scale 0.7 --height-scan --steps 500_000 `
    --eval-freq 50_000 --eval-episodes 20 --save-dir models/go2_ppo_flat

python -m go2_ppo.train --terrain steps --action-scale 0.7 --height-scan --steps 500_000 `
    --eval-freq 50_000 --eval-episodes 20 --save-dir models/go2_ppo_steps

python -m go2_ppo.train --terrain full --action-scale 0.7 --height-scan --steps 500_000 `
    --eval-freq 50_000 --eval-episodes 20 --save-dir models/go2_ppo_full