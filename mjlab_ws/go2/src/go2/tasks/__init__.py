from __future__ import annotations

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import go2_env_cfg
from .rl_cfg import go2_ppo_runner_cfg


def _register(task_id: str, terrain: str) -> None:
    register_mjlab_task(
        task_id=task_id,
        env_cfg=go2_env_cfg(terrain=terrain, height_scan=True),
        play_env_cfg=go2_env_cfg(
            terrain=terrain,
            height_scan=True,
            num_envs=1,
            play=True,
            # Evaluation should reproduce the trained policy deterministically;
            # turn DR off unless the user overrides --env.dr-scale when playing.
            dr_scale=0.0,
        ),
        rl_cfg=go2_ppo_runner_cfg(experiment_name=f"go2_ppo_{terrain}"),
        runner_cls=MjlabOnPolicyRunner,
    )


_register("Mjlab-Go2", "full")
_register("Mjlab-Go2-Full", "full")
_register("Mjlab-Go2-Flat", "flat")
_register("Mjlab-Go2-Steps", "steps")
