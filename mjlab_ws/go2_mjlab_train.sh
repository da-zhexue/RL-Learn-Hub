uv run python -m mjlab.scripts.train Mjlab-Go2-Flat --log-root go2/logs/flat --video True \
    --agent.max-iterations 5000 --agent.save-interval 200 --env.scene.num-envs 4096 \
    --agent.logger tensorboard 
