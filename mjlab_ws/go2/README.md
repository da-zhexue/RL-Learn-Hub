# Go2 PPO on mjlab

This package ports the existing `go2_ppo` task into mjlab. It keeps the original
47-dimensional proprioceptive observation, heading and command layout, 50 Hz
control rate, PD law, terrain course, termination rules, and all 18 reward terms
from `go2_common/config.py`.

## Register and train

From `mjlab_ws`:

```bash
uv sync
uv run list-envs
uv run python -m mjlab.scripts.train Mjlab-Go2 \
  --agent.max-iterations 1465 --log-root logs/go2_ppo
```

Available tasks:

- `Mjlab-Go2` / `Mjlab-Go2-Full`: full course (`full`, scale 1.0)
- `Mjlab-Go2-Steps`: two low obstacles
- `Mjlab-Go2-Flat`: flat ground

Playback:

```bash
uv run python -m mjlab.scripts.play Mjlab-Go2 \
  --checkpoint_file <checkpoint.pt> --num_envs 1 --viewer viser
```

The default PPO configuration intentionally mirrors `go2_ppo`:

- 8 environments, 256 rollout steps, 3 epochs, 8 mini-batches
- learning rate `1e-4`
- gamma `0.995`, GAE lambda `0.95`
- clip range `0.2`, gradient norm `0.5`
- initial Gaussian std `exp(-1)` (same as `log_std_init=-1`)
- action clipping at `+-1`

The current Windows machine was not used to run mjlab; run the commands above on
the target machine.
