from __future__ import annotations

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def go2_ppo_runner_cfg(experiment_name: str = "go2_ppo") -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.36787944117144233,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=0.5,
            use_clipped_value_loss=False,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=8,
            learning_rate=1.0e-4,
            schedule="fixed",
            gamma=0.995,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=0.5,
        ),
        experiment_name=experiment_name,
        logger="tensorboard",
        save_interval=98,
        num_steps_per_env=256,
        max_iterations=1465,
        clip_actions=1.0,
    )
