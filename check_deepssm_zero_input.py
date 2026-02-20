"""Quick diagnostic for DeepSSM zero-input behavior.

This script checks whether DeepSSM outputs go to zero when input goes to zero,
and lets you force all Linear biases to zero for testing.

Examples:
  python check_deepssm_zero_input.py --param lru --ff LGLU
  python check_deepssm_zero_input.py --param lru --ff LGLU --zero_bias
  python check_deepssm_zero_input.py --param l2n --ff LMLP --mode loop
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DeepSSM zero-input diagnostic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--param", type=str, default="lru")
    p.add_argument("--ff", type=str, default="LGLU")
    p.add_argument("--mode", type=str, default="loop", choices=["loop", "scan"])
    p.add_argument("--horizon", type=int, default=200)

    p.add_argument("--d_input", type=int, default=4)
    p.add_argument("--d_output", type=int, default=16)
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--d_state", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--zero_bias", action="store_true", help="Force all nn.Linear biases to zero.")
    p.add_argument("--freeze_bias", action="store_true", help="Also set requires_grad=False on those biases.")

    return p.parse_args()


def linear_bias_report(module: nn.Module) -> List[Tuple[str, float]]:
    report: List[Tuple[str, float]] = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            report.append((name, float(m.bias.detach().abs().max().item())))
    return report


def force_zero_linear_bias(module: nn.Module, freeze: bool = False) -> int:
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            with torch.no_grad():
                m.bias.zero_()
            if freeze:
                m.bias.requires_grad_(False)
            count += 1
    return count


def run_full_sequence(model: nn.Module, w: torch.Tensor, mode: str) -> torch.Tensor:
    with torch.no_grad():
        y, _ = model(w, mode=mode, reset_state=True, detach_state=True)
    return y


def run_stepwise(model: nn.Module, w: torch.Tensor, mode: str) -> torch.Tensor:
    # Emulates rollout usage with T=1 calls and persistent internal state.
    model.reset()
    ys = []
    state = None
    with torch.no_grad():
        for t in range(w.shape[1]):
            y_t, state = model(
                w[:, t:t + 1, :],
                state=state,
                mode=mode,
                reset_state=False,
                detach_state=True,
            )
            ys.append(y_t)
    return torch.cat(ys, dim=1)


def summarize(y: torch.Tensor, label: str) -> None:
    n = torch.norm(y, dim=-1).squeeze(0)
    print(
        f"{label:>16s} | first={float(n[0].item()):.4e} "
        f"tail={float(n[-1].item()):.4e} min={float(n.min().item()):.4e} max={float(n.max().item()):.4e}"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from neural_ssm import DeepSSM

    model = DeepSSM(
        args.d_input,
        args.d_output,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        dropout=args.dropout,
        param=args.param,
        ff=args.ff,
    )

    print("Config:", vars(args))

    before = linear_bias_report(model)
    if before:
        print(f"Linear bias tensors before: {len(before)}")
        print("max |bias| (top 8):", [f"{n}:{v:.2e}" for n, v in before[:8]])
    else:
        print("Linear bias tensors before: 0")

    if args.zero_bias:
        n_zeroed = force_zero_linear_bias(model, freeze=args.freeze_bias)
        print(f"Zeroed linear biases: {n_zeroed} (freeze={args.freeze_bias})")
        after = linear_bias_report(model)
        if after:
            print("max |bias| after (top 8):", [f"{n}:{v:.2e}" for n, v in after[:8]])

    T = args.horizon
    w_zero = torch.zeros(1, T, args.d_input)
    w_imp = torch.zeros(1, T, args.d_input)
    w_imp[:, 0, : min(args.d_input, 4)] = torch.tensor([1.0, -0.5, 0.3, 0.0])[: min(args.d_input, 4)]

    y_zero_full = run_full_sequence(model, w_zero, args.mode)
    y_imp_full = run_full_sequence(model, w_imp, args.mode)

    y_zero_step = run_stepwise(model, w_zero, args.mode)
    y_imp_step = run_stepwise(model, w_imp, args.mode)

    print("\nFull-sequence results")
    summarize(y_zero_full, "zero input")
    summarize(y_imp_full, "impulse")

    print("\nStepwise T=1 results")
    summarize(y_zero_step, "zero input")
    summarize(y_imp_step, "impulse")

    diff_zero = float((y_zero_full - y_zero_step).abs().max().item())
    diff_imp = float((y_imp_full - y_imp_step).abs().max().item())
    print(f"\nmax |full-stepwise| zero input: {diff_zero:.3e}")
    print(f"max |full-stepwise| impulse:    {diff_imp:.3e}")


if __name__ == "__main__":
    main()
