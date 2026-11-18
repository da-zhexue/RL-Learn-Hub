from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg, mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from go2.asset_zoo.robots.go2_constants import get_go2_robot_cfg

from . import rewards as reward_fns
from . import state, terminations
from .actions import Go2JointPositionActionCfg
from .commands import CourseVelocityCommandCfg
from .config import EnvCfg, RewardCfg
from .context import FOOT_GEOM_NAMES
from .dr_cfg import build_dr_events
from .observations import policy_obs
from .terms import TERM_NAMES
from .terrain import course_spec_fn


@dataclass(kw_only=True)
class Go2EnvCfg(ManagerBasedRlEnvCfg):
    """Go2 task config with the terrain/observation switches context.py needs.

    These must be real dataclass fields: ``train.py`` rebuilds the config through
    tyro, which drops attributes assigned dynamically after construction.
    """

    terrain_variant: str = "full"
    terrain_scale: float = 1.0
    privileged: bool = False
    height_scan: bool = True
    dr_scale: float = 1.0
    """Domain randomization amplitude. 0 disables DR entirely, 1.0 uses the
    full ranges in ``dr_cfg.py``."""

    def __post_init__(self) -> None:
        # Rebuild the course callback from the final field values: train.py parses
        # the config with tyro, which reconstructs it (calling __post_init__) and
        # may override terrain_variant/terrain_scale on the command line.
        self.scene.spec_fn = course_spec_fn(self.terrain_variant, self.terrain_scale)
        # Same reason as spec_fn: tyro reconstructs the config and may override
        # dr_scale on the command line, so the DR event terms are rebuilt from
        # the final value here instead of once in go2_env_cfg().
        for name in [name for name in self.events if name.startswith("dr_")]:
            del self.events[name]
        self.events.update(build_dr_events(self.dr_scale))


def go2_env_cfg(
    *,
    terrain: str = "full",
    terrain_scale: float = 1.0,
    privileged: bool = False,
    height_scan: bool = True,
    num_envs: int = 8,
    play: bool = False,
    dr_scale: float = 1.0,
) -> Go2EnvCfg:
    ref = EnvCfg(terrain=terrain, terrain_scale=terrain_scale, privileged=privileged, height_scan=height_scan)
    reward_cfg = RewardCfg()
    # mjWarp replicates one model into every world, so each world already gets its
    # own copy of the single course. Distinct env origins would only shift robots
    # in that shared model; all-zero origins keep every env on the course.
    env_spacing = 0.0

    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    body_ground = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(mode="geom", pattern=(".*_collision_.*",), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    actor_terms = {
        "policy": ObservationTermCfg(func=policy_obs),
    }
    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joints": Go2JointPositionActionCfg(
            entity_name="robot",
            scale=ref.action_scale,
            kp=ref.kp,
            kd=ref.kd,
            joint_margin=ref.joint_margin,
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "course_velocity": CourseVelocityCommandCfg(
            resampling_time_range=(1e9, 1e9),
            vx_range=ref.cmd_vx,
        )
    }
    events = {
        "reset_base": EventTermCfg(func=state.reset_base, mode="reset"),
        "reset_prev": EventTermCfg(func=state.reset_prev, mode="reset"),
        "advance_prev": EventTermCfg(func=state.advance_prev, mode="step"),
    }

    rewards = {
        name: RewardTermCfg(
            func=getattr(reward_fns, f"reward_{name}"),
            weight=float(getattr(reward_cfg, name)),
        )
        for name in TERM_NAMES
    }
    terminations_cfg = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "success": TerminationTermCfg(func=terminations.success),
        "fallen": TerminationTermCfg(func=terminations.fallen),
        "flipped": TerminationTermCfg(func=terminations.flipped),
        "out_of_course": TerminationTermCfg(func=terminations.out_of_course),
    }

    scene = SceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        extent=10.0,
        terrain=TerrainEntityCfg(
            terrain_type="plane",
            env_spacing=env_spacing,
        ),
        entities={"robot": get_go2_robot_cfg()},
        sensors=(feet_ground, body_ground),
    )

    cfg = Go2EnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations_cfg,
        sim=SimulationCfg(
            # Per-world capacities: memory is nconmax * njmax * num_envs. 4096
            # envs with 500/1500 needed >24 GiB; 128/384 keeps a quadruped's
            # contacts/constraints comfortable at a fraction of the footprint.
            nconmax=128,
            njmax=384,
            contact_sensor_maxmatch=512,
            mujoco=MujocoCfg(
                timestep=ref.sim_dt,
                integrator="euler",
                cone="elliptic",
                impratio=100.0,
                iterations=100,
                ls_iterations=20,
                ccd_iterations=50,
            ),
        ),
        decimation=ref.decimation,
        episode_length_s=1e9 if play else ref.max_episode_s,
        scale_rewards_by_dt=False,
        is_finite_horizon=False,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=3.0,
            elevation=-10.0,
            azimuth=90.0,
            # All env origins coincide now, so "neighbor" envs would render on
            # top of the tracked robot.
            max_extra_envs=0,
        ),
        terrain_variant=terrain,
        terrain_scale=terrain_scale,
        privileged=privileged,
        height_scan=height_scan,
        dr_scale=dr_scale,
    )
    return cfg
