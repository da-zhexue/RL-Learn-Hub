from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .config import DEFAULT_Q
from .context import ACTUATOR_NAMES, JOINT_NAMES


@dataclass(kw_only=True)
class Go2JointPositionActionCfg(ActionTermCfg):
    joint_names: tuple[str, ...] = JOINT_NAMES
    scale: float = 0.6
    kp: float = 60.0
    kd: float = 3.5
    joint_margin: float = 0.05

    def build(self, env) -> "Go2JointPositionAction":
        return Go2JointPositionAction(self, env)


class Go2JointPositionAction(ActionTerm):
    """Explicit per-substep PD torque law matching go2_ppo's env.step()."""

    cfg: Go2JointPositionActionCfg

    def __init__(self, cfg: Go2JointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._joint_ids, names = self._entity.find_joints(
            list(cfg.joint_names), preserve_order=True
        )
        if tuple(names) != tuple(cfg.joint_names):
            raise RuntimeError(f"joint order mismatch: requested={cfg.joint_names}, got={tuple(names)}")

        self._actuator_ids, actuator_names = self._entity.find_actuators(
            list(ACTUATOR_NAMES), preserve_order=True
        )
        if tuple(actuator_names) != ACTUATOR_NAMES:
            raise RuntimeError(
                f"actuator order mismatch: requested={ACTUATOR_NAMES}, got={tuple(actuator_names)}"
            )

        limits = self._entity.data.joint_pos_limits[0][self._joint_ids].to(torch.float64)
        self._q_lo = limits[:, 0] + cfg.joint_margin
        self._q_hi = limits[:, 1] - cfg.joint_margin
        self._q_default = torch.clamp(
            torch.tensor(DEFAULT_Q, dtype=torch.float64, device=env.device),
            self._q_lo,
            self._q_hi,
        )

        global_ctrl_ids = self._entity.indexing.ctrl_ids[
            torch.tensor(self._actuator_ids, dtype=torch.long, device=env.device)
        ]
        ctrlrange = env.sim.mj_model.actuator_ctrlrange[global_ctrl_ids.cpu().numpy()]
        self._tau_lo = torch.as_tensor(ctrlrange[:, 0], dtype=torch.float64, device=env.device)
        self._tau_hi = torch.as_tensor(ctrlrange[:, 1], dtype=torch.float64, device=env.device)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=env.device)
        self._target = self._q_default.unsqueeze(0).repeat(self.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._target = torch.clamp(
            self._q_default + self.cfg.scale * self._raw_actions,
            min=self._q_lo,
            max=self._q_hi,
        )

    def apply_actions(self) -> None:
        q = self._entity.data.joint_pos[:, self._joint_ids].to(torch.float64)
        dq = self._entity.data.joint_vel[:, self._joint_ids].to(torch.float64)
        tau = self.cfg.kp * (self._target - q) - self.cfg.kd * dq
        tau = torch.clamp(tau, min=self._tau_lo, max=self._tau_hi)
        self._entity.set_joint_effort_target(
            tau.to(torch.float32),
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._target[env_ids] = self._q_default
