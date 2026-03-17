"""Nominal and true dynamics for 2D navigation (pre-stabilized double integrator)."""

from __future__ import annotations

from typing import Optional

import torch

from pb_core import as_bt


class DoubleIntegratorNominal:
    """
    Pre-stabilized model around the origin.

    Closed-loop acceleration:
      a = -k_p * pos - k_d * vel + u
    where u is the PB boost input.
    """

    def __init__(self, dt: float = 0.05, pre_kp: float = 1.0, pre_kd: float = 1.5):
        self.dt = dt
        self.pre_kp = pre_kp
        self.pre_kd = pre_kd

    def _pre_stab_acc(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        return -self.pre_kp * pos - self.pre_kd * vel

    def nominal_dynamics(self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        x = as_bt(x)
        u = as_bt(u)
        pos = x[..., :2]
        vel = x[..., 2:]

        pos_next = pos + self.dt * vel
        acc = self._pre_stab_acc(pos, vel) + u
        vel_next = vel + self.dt * acc
        return torch.cat([pos_next, vel_next], dim=-1)


class DoubleIntegratorTrue:
    """Perfect model: true dynamics equal nominal dynamics (no disturbances)."""

    def __init__(self, dt: float = 0.05, pre_kp: float = 1.0, pre_kd: float = 1.5):
        self.dt = dt
        self.pre_kp = pre_kp
        self.pre_kd = pre_kd

    def _pre_stab_acc(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        return -self.pre_kp * pos - self.pre_kd * vel

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        x = as_bt(x)
        u = as_bt(u)
        pos = x[..., :2]
        vel = x[..., 2:]

        pos_next = pos + self.dt * vel
        acc = self._pre_stab_acc(pos, vel) + u
        vel_next = vel + self.dt * acc
        return torch.cat([pos_next, vel_next], dim=-1)
