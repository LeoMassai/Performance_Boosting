"""
BPTT rollout utilities and a minimal demo for the PBController.

Run:
  python rollout_bptt.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from pb_controller import (
    PBController,
    LinearPlant,
    FactorizedOperator,
    MLP,
    MbMLP,
    as_bt,
)


class TruePlant:
    """Simple nonlinear true plant used for BPTT demo."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, nonlin_scale: float = 0.05):
        self.A = A
        self.B = B
        self.nonlin_scale = nonlin_scale

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        x = as_bt(x)
        u = as_bt(u)
        lin = torch.einsum("ij,btj->bti", self.A, x) + torch.einsum("ij,btj->bti", self.B, u)
        nonlin = self.nonlin_scale * torch.tanh(x)
        return lin + nonlin


def rollout_bptt(
    controller: PBController,
    plant_true: TruePlant,
    x0: torch.Tensor,
    horizon: int,
    z_seq: Optional[torch.Tensor] = None,
    context_fn: Optional[callable] = None,
    w0: Optional[torch.Tensor] = None,
    u_post_fn: Optional[callable] = None,
    process_noise_seq: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Causal rollout with BPTT.

    Provide either:
      - z_seq: (B, T, Nz) precomputed exogenous context, or
      - context_fn: callable z_t = context_fn(x_t, t) for state-dependent context.
    """
    if (z_seq is None) == (context_fn is None):
        raise ValueError("Provide exactly one of z_seq or context_fn.")

    x0 = as_bt(x0)
    z_seq = as_bt(z_seq) if z_seq is not None else None
    if process_noise_seq is not None:
        process_noise_seq = as_bt(process_noise_seq)
        if process_noise_seq.shape[0] != x0.shape[0] or process_noise_seq.shape[1] != horizon:
            raise ValueError(
                "process_noise_seq must have shape (B, horizon, Nx). "
                f"Got {tuple(process_noise_seq.shape)} with B={x0.shape[0]}, horizon={horizon}."
            )
        if process_noise_seq.shape[-1] != x0.shape[-1]:
            raise ValueError(
                "process_noise_seq last dim must match state dim. "
                f"Got {process_noise_seq.shape[-1]} vs {x0.shape[-1]}."
            )

    controller.reset(x0, u_init=None, w0=w0)

    x = x0
    x_log = []
    u_log = []
    w_log = []

    for t in range(horizon):
        if z_seq is not None:
            z_t = z_seq[:, t:t + 1, :]
        else:
            z_t = context_fn(x, t)

        u_t, w_t = controller.forward_step(x, z_t, t=t)
        if u_post_fn is not None:
            # Backward-compatible post-processing API:
            #   u_post_fn(u) or u_post_fn(x, u, t)
            try:
                u_applied = u_post_fn(x, u_t, t)
            except TypeError:
                u_applied = u_post_fn(u_t)
            controller.set_last_applied_control(u_applied)
            u_t = u_applied
        x = plant_true.forward(x, u_t, t=t)
        if process_noise_seq is not None:
            x = x + process_noise_seq[:, t:t + 1, :]

        x_log.append(x)
        u_log.append(u_t)
        w_log.append(w_t)

    return torch.cat(x_log, dim=1), torch.cat(u_log, dim=1), torch.cat(w_log, dim=1)


def demo() -> None:
    torch.manual_seed(0)

    B, T = 8, 25
    Nx, Nu, Nz = 4, 2, 3
    s = 8

    A = torch.eye(Nx)
    Bmat = torch.randn(Nx, Nu) * 0.1

    nominal = LinearPlant(A, Bmat)
    true_plant = TruePlant(A, Bmat, nonlin_scale=0.05)

    mp = MLP(Nx, s)
    mb = MbMLP(Nx + Nz, r=Nu, s=s)
    op = FactorizedOperator(mp, mb)

    def u_nominal(x, t=None):
        return torch.zeros(x.shape[0], x.shape[1], Nu, device=x.device)

    controller = PBController(
        plant=nominal,
        operator=op,
        u_nominal=u_nominal,
        u_dim=Nu,
        detach_state=False,  # enable BPTT through controller state
    )

    x0 = torch.randn(B, 1, Nx) * 0.1
    z_seq = torch.randn(B, T, Nz)

    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)

    for step in range(5):
        optimizer.zero_grad()
        x_seq, u_seq, _ = rollout_bptt(
            controller=controller,
            plant_true=true_plant,
            x0=x0,
            horizon=T,
            z_seq=z_seq,
        )

        pos = x_seq[:, :, :2]
        loss = (pos ** 2).mean() + 0.01 * (u_seq ** 2).mean()
        loss.backward()
        optimizer.step()

        print(f"step {step:02d} | loss {loss.item():.6f}")


if __name__ == "__main__":
    demo()
