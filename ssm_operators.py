"""Adapters for using neural_ssm modules as PB operators."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pb_core import as_bt


class MpDeepSSM(nn.Module):
    """
    Adapter for M_p(w) implemented with neural_ssm.DeepSSM.

    Input:
      w: (B, T, w_dim) or (B, w_dim)
    Output:
      v: (B, T, s_dim)
    """

    def __init__(
        self,
        w_dim: int,
        s_dim: int,
        *,
        mode: str = "loop",
        reset_state_each_call: bool = False,
        detach_state: bool = False,
        **ssm_kwargs: Any,
    ):
        super().__init__()
        try:
            from neural_ssm import DeepSSM
        except ImportError as exc:
            raise ImportError(
                "Could not import neural_ssm.DeepSSM. Install neural_ssm in this environment."
            ) from exc

        self.core = DeepSSM(w_dim, s_dim, **ssm_kwargs)
        self.mode = mode
        self.reset_state_each_call = reset_state_each_call
        self.detach_state = detach_state

    def reset(self) -> None:
        if hasattr(self.core, "reset"):
            self.core.reset()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w = as_bt(w)
        out = self.core(
            w,
            mode=self.mode,
            reset_state=self.reset_state_each_call,
            detach_state=self.detach_state,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = as_bt(out)
        return out
