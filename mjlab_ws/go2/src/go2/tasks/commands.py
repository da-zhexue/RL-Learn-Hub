from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class CourseVelocityCommandCfg(CommandTermCfg):
    vx_range: tuple[float, float] = (0.4, 0.8)

    def build(self, env) -> "CourseVelocityCommand":
        return CourseVelocityCommand(self, env)


class CourseVelocityCommand(CommandTerm):
    """Per-episode forward command; there is no periodic resampling."""

    cfg: CourseVelocityCommandCfg

    def __init__(self, cfg: CourseVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        lo, hi = self.cfg.vx_range
        vx = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
        self._command[env_ids, 0] = vx
        self._command[env_ids, 1] = 0.0
        self._command[env_ids, 2] = 0.0

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
