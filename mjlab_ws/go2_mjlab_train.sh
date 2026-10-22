# 阶段 1：平地
uv run python -m mjlab.scripts.train Mjlab-Go2-Flat \
    --log-root go2/logs/ppo \
    --agent.experiment-name go2_ppo \
    --agent.run-name flat \
    --agent.max-iterations 200 \
    --agent.save-interval 20 \
    --agent.logger tensorboard \
    --env.scene.num-envs 4096 \
    --env.terrain-variant flat \
    --env.terrain-scale 1.0 \
    --env.height-scan True \
    --env.actions.joints.scale 0.7 \
    --agent.algorithm.entropy-coef 0.005 \
    --video True

# 阶段 2：两级矮台阶
uv run python -m mjlab.scripts.train Mjlab-Go2-Steps \
    --log-root go2/logs/ppo \
    --agent.experiment-name go2_ppo \
    --agent.run-name steps \
    --agent.max-iterations 150 \
    --agent.save-interval 18 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run '.*_flat' \
    --agent.load-checkpoint 'model_.*\.pt' \
    --env.scene.num-envs 4096 \
    --env.terrain-variant steps \
    --env.terrain-scale 1.0 \
    --env.height-scan True \
    --env.actions.joints.scale 0.7 \
    --agent.algorithm.entropy-coef 0.003 \
    --video True

# 阶段 3.1：full 0.4
uv run python -m mjlab.scripts.train Mjlab-Go2-Full \
    --log-root go2/logs/ppo \
    --agent.experiment-name go2_ppo \
    --agent.run-name full_s04 \
    --agent.max-iterations 150 \
    --agent.save-interval 18 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run '.*_steps' \
    --agent.load-checkpoint 'model_.*\.pt' \
    --env.scene.num-envs 4096 \
    --env.terrain-variant full \
    --env.terrain-scale 0.4 \
    --env.height-scan True \
    --env.actions.joints.scale 0.7 \
    --video True

# 阶段 3.2：full 0.6
uv run python -m mjlab.scripts.train Mjlab-Go2-Full \
    --log-root go2/logs/ppo \
    --agent.experiment-name go2_ppo \
    --agent.run-name full_s06 \
    --agent.max-iterations 150 \
    --agent.save-interval 18 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run '.*_full_s04' \
    --agent.load-checkpoint 'model_.*\.pt' \
    --env.scene.num-envs 4096 \
    --env.terrain-variant full \
    --env.terrain-scale 0.6 \
    --env.height-scan True \
    --env.actions.joints.scale 0.7 \
    --video True

# 阶段 3.3：full 0.8
uv run python -m mjlab.scripts.train Mjlab-Go2-Full \
    --log-root go2/logs/ppo \
    --agent.experiment-name go2_ppo \
    --agent.run-name full_s08 \
    --agent.max-iterations 150 \
    --agent.save-interval 18 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run '.*_full_s06' \
    --agent.load-checkpoint 'model_.*\.pt' \
    --env.scene.num-envs 4096 \
    --env.terrain-variant full \
    --env.terrain-scale 0.8 \
    --env.height-scan True \
    --env.actions.joints.scale 0.7 \
    --video True

# 阶段 3.4：full 1.0
uv run python -m mjlab.scripts.train Mjlab-Go2-Full \
    --log-root go2/logs/ppo \
    --agent.experiment-name go2_ppo \
    --agent.run-name full_s10 \
    --agent.max-iterations 200 \
    --agent.save-interval 25 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run '.*_full_s08' \
    --agent.load-checkpoint 'model_.*\.pt' \
    --env.scene.num-envs 4096 \
    --env.terrain-variant full \
    --env.terrain-scale 1.0 \
    --env.height-scan True \
    --env.actions.joints.scale 0.7 \
    --video True

# 阶段 4.1：修复，回 full 0.9
uv run python -m mjlab.scripts.train Mjlab-Go2-Full \
    --log-root go2/logs/ppo \
    --agent.experiment-name go2_ppo \
    --agent.run-name repair_s09 \
    --agent.max-iterations 200 \
    --agent.save-interval 25 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run '.*_full_s10' \
    --agent.load-checkpoint 'model_.*\.pt' \
    --env.scene.num-envs 4096 \
    --env.terrain-variant full \
    --env.terrain-scale 0.9 \
    --env.height-scan True \
    --env.actions.joints.scale 0.7 \
    --agent.algorithm.entropy-coef 0.003 \
    --video True

# 阶段 4.2：修复，回 full 1.0 并重排奖励
uv run python -m mjlab.scripts.train Mjlab-Go2-Full \
    --log-root go2/logs/ppo \
    --agent.experiment-name go2_ppo \
    --agent.run-name repair_s10 \
    --agent.max-iterations 200 \
    --agent.save-interval 25 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run '.*_repair_s09' \
    --agent.load-checkpoint 'model_.*\.pt' \
    --env.scene.num-envs 4096 \
    --env.terrain-variant full \
    --env.terrain-scale 1.0 \
    --env.height-scan True \
    --env.actions.joints.scale 0.7 \
    --agent.algorithm.entropy-coef 0.003 \
    --env.rewards.feet-clearance.weight 0.5 \
    --env.rewards.feet-air-time.weight 2 \
    --env.rewards.foot-swing-height.weight 2 \
    --env.rewards.climb.weight 800 \
    --env.rewards.level-bonus.weight 60 \
    --env.rewards.orientation.weight -2.0 \
    --video True

# 回放最后训好的模型
uv run python -m mjlab.scripts.play Mjlab-Go2-Full \
    --checkpoint-file "$(ls -t go2/logs/ppo/go2_ppo/*_repair_s10/model_*.pt | head -1)" \
    --num-envs 4 \
    --viewer viser
