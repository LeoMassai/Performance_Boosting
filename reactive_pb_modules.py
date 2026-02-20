"""
Fresh operator modules for context-reactive PB control.

This module defines:
  - StableLatentGRUOperator for M_p(w)
  - ContextReactiveBoundedOperator for M_b(w, z)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from pb_controller import as_bt


def _bounded_map(x: torch.Tensor, mode: str, bound_value: float) -> torch.Tensor:
    if bound_value <= 0:
        raise ValueError(f"bound_value must be > 0, got {bound_value}")

    if mode == "tanh":
        return bound_value * torch.tanh(x / bound_value)
    if mode == "softsign":
        return bound_value * (x / (1.0 + torch.abs(x)))
    if mode == "clamp":
        return torch.clamp(x, min=-bound_value, max=bound_value)
    raise ValueError(f"Unknown bound mode: {mode!r}")


def _make_spectral_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    if num_layers < 2:
        raise ValueError("num_layers must be >= 2")

    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(spectral_norm(nn.Linear(dims[i], dims[i + 1])))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class StableLatentGRUOperator(nn.Module):
    """
    Stateful latent map M_p(w) with bounded output.

    Input:
      w: (B, T, w_dim) or (B, w_dim)
    Output:
      v: (B, T, s_dim)
    """

    def __init__(
        self,
        w_dim: int,
        s_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bound_mode: str = "softsign",
        bound_value: float = 8.0,
        detach_state: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.w_dim = int(w_dim)
        self.s_dim = int(s_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bound_mode = str(bound_mode)
        self.bound_value = float(bound_value)
        self.detach_state = bool(detach_state)

        self.gru = nn.GRU(
            input_size=self.w_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.proj = spectral_norm(nn.Linear(self.hidden_dim, self.s_dim))
        self._hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._hidden = None

    def _prepare_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._hidden is None:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        h = self._hidden
        if h.shape[1] != batch_size or h.device != device or h.dtype != dtype:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        return h

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w = as_bt(w)
        if w.shape[-1] != self.w_dim:
            raise ValueError(f"w last dim must be {self.w_dim}, got {w.shape[-1]}")

        bsz, t_steps, _ = w.shape
        h0 = self._prepare_hidden(bsz, w.device, w.dtype)
        out, h_n = self.gru(w, h0)

        raw = self.proj(out.reshape(bsz * t_steps, self.hidden_dim)).view(bsz, t_steps, self.s_dim)
        v = _bounded_map(raw, self.bound_mode, self.bound_value)

        self._hidden = h_n.detach() if self.detach_state else h_n
        return v


class ContextReactiveBoundedOperator(nn.Module):
    """
    Bounded, context-amplified M_b(w, z).

    The construction mixes:
      - a base branch driven by [w, z],
      - a direct context branch,
      - a context gate that multiplicatively amplifies sensitivity,
      - a cross branch on elementwise w-z interactions.
    """

    def __init__(
        self,
        w_dim: int,
        z_dim: int,
        r: int,
        s: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        bound_mode: str = "softsign",
        bound_value: float = 25.0,
        ctx_gain: float = 4.0,
        ctx_direct_gain: float = 3.0,
        ctx_cross_gain: float = 1.5,
        ctx_gate_gain: float = 8.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.w_dim = int(w_dim)
        self.z_dim = int(z_dim)
        self.r = int(r)
        self.s = int(s)
        self.bound_mode = str(bound_mode)
        self.bound_value = float(bound_value)
        self.ctx_gain = float(ctx_gain)
        self.ctx_direct_gain = float(ctx_direct_gain)
        self.ctx_cross_gain = float(ctx_cross_gain)
        self.ctx_gate_gain = float(ctx_gate_gain)

        self.w_norm = nn.LayerNorm(self.w_dim)
        self.z_norm = nn.LayerNorm(self.z_dim)
        self.z_to_w = spectral_norm(nn.Linear(self.z_dim, self.w_dim))

        mix_dim = self.w_dim + self.z_dim + self.w_dim
        out_dim = self.r * self.s

        self.base_net = _make_spectral_mlp(
            in_dim=mix_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=nn.SiLU,
        )
        self.ctx_net = _make_spectral_mlp(
            in_dim=self.z_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=max(2, num_layers - 1),
            activation=nn.SiLU,
        )
        self.gate_net = _make_spectral_mlp(
            in_dim=self.z_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=max(2, num_layers - 1),
            activation=nn.SiLU,
        )
        self.cross_net = _make_spectral_mlp(
            in_dim=mix_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=max(2, num_layers - 1),
            activation=nn.SiLU,
        )

    def reset(self) -> None:
        return None

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is None:
            raise ValueError("ContextReactiveBoundedOperator requires z.")

        w = as_bt(w)
        z = as_bt(z)

        if w.shape[:2] != z.shape[:2]:
            raise ValueError(f"w and z leading dims must match. Got {tuple(w.shape)} vs {tuple(z.shape)}")
        if w.shape[-1] != self.w_dim:
            raise ValueError(f"w last dim must be {self.w_dim}, got {w.shape[-1]}")
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"z last dim must be {self.z_dim}, got {z.shape[-1]}")

        bsz, t_steps, _ = w.shape
        w_flat = self.w_norm(w.reshape(bsz * t_steps, self.w_dim))
        z_flat = self.z_norm(z.reshape(bsz * t_steps, self.z_dim))

        z_proj = torch.tanh(self.z_to_w(z_flat))
        wz = w_flat * z_proj
        mix = torch.cat([w_flat, z_flat, wz], dim=-1)

        base = self.base_net(mix)
        ctx_direct = self.ctx_net(z_flat)
        gate = torch.sigmoid(self.ctx_gate_gain * self.gate_net(z_flat))
        cross = torch.tanh(self.cross_net(mix))

        raw = (
            base * (1.0 + self.ctx_gain * gate)
            + self.ctx_direct_gain * ctx_direct
            + self.ctx_cross_gain * cross
        )
        bounded = _bounded_map(raw, self.bound_mode, self.bound_value)
        return bounded.view(bsz, t_steps, self.r, self.s)
