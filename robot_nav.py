"""
PB navigation training script with validation/checkpointing and plots.

Fixes compared to the original draft:
  - Uses w0=x0 in rollout (PB disturbance convention).
  - Moves sampled scenarios to device once (no per-step CPU->GPU context transfers).
  - Tracks validation metrics and saves best checkpoint.
  - Implements Corridor Penalty to prevent "squeezing" through impassable gaps.
  - Explicitly curates "right side" challenges to force learning the long-way around.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from bounded_mlp_operator import BoundedMLPOperator
from nav_env import (
    collision_penalty,
    collision_violation_penalty,
    direct_path_intersection_mask,
    ensure_challenging_starts,
    min_dist_to_edge,
    sample_scenarios,
    build_context,
)
from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from pb_controller import PBController, FactorizedOperator
from rollout_bptt import rollout_bptt
from ssm_operators import MpDeepSSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PB robot navigation training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--val_batch", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="/Users/leo/Library/CloudStorage/GoogleDrive-l.massai@epfl.ch/My Drive/Code/Projects/Performance_Boosting/runs/nav_experiment/robot_nav_20260224_124406",
        help='Checkpoint to load before training: path, "auto_latest_best", or "" to disable',
    )
    parser.add_argument("--no_show_plots", action="store_true")
    parser.add_argument("--start_box", type=float, default=3.0)
    parser.add_argument("--heatmap_res", type=int, default=140)
    parser.add_argument("--heatmap_batch", type=int, default=4096)
    parser.add_argument("--radius_cmp_num_starts", type=int, default=4)
    parser.add_argument("--radius_cmp_anchor_x", type=float, default=2.0)
    parser.add_argument("--radius_cmp_anchor_y", type=float, default=0.0)
    parser.add_argument("--radius_cmp_margin", type=float, default=0.05)

    # Loss weights
    parser.add_argument("--w_term", type=float, default=30.0)
    parser.add_argument("--w_stage", type=float, default=1.0)
    parser.add_argument("--w_coll_soft", type=float, default=40.0)
    parser.add_argument("--w_coll_hard", type=float, default=100.0)
    parser.add_argument("--w_corridor", type=float, default=50.0)  # ACTIVATED
    parser.add_argument("--w_control", type=float, default=0.05)

    # Collision & Corridor penalty shape
    parser.add_argument("--coll_margin", type=float, default=0.1)
    parser.add_argument("--coll_beta", type=float, default=10.0)
    parser.add_argument("--corridor_gap_crit", type=float, default=0.08)
    parser.add_argument("--corridor_margin", type=float, default=0.03)
    parser.add_argument("--corridor_beta", type=float, default=12.0)
    parser.add_argument("--corridor_gap_beta", type=float, default=20.0)

    # Process noise used in training rollout
    parser.add_argument("--noise_sigma0", type=float, default=0.05)
    parser.add_argument("--noise_tau", type=float, default=20.0)

    # Obstacles and curriculum
    parser.add_argument("--k_obstacles", type=int, default=3)
    parser.add_argument("--challenge_frac", type=float, default=0.68)
    parser.add_argument("--right_challenge_frac_train", type=float, default=0.25)
    parser.add_argument("--right_challenge_frac_val", type=float, default=0.35)
    parser.add_argument("--right_x_min", type=float, default=0.5)
    parser.add_argument("--right_challenge_margin", type=float, default=0.05)
    parser.add_argument("--curriculum_epochs", type=int, default=100)
    parser.add_argument("--r_min_start", type=float, default=0.1)
    parser.add_argument("--r_min_end", type=float, default=0.2)
    parser.add_argument("--r_max_start", type=float, default=0.2)
    parser.add_argument("--r_max_end", type=float, default=0.8)

    parser.add_argument(
        "--best_ckpt_metric",
        type=str,
        default="right_collision_then_loss",  # DEFAULT TO THE NEW METRIC
        choices=["loss", "collision_then_loss", "right_collision_then_loss"],
    )
    parser.add_argument("--best_ckpt_collision_tol", type=float, default=1e-4)
    return parser.parse_args()


def _find_latest_best_checkpoint(runs_root: str) -> str | None:
    if not os.path.isdir(runs_root):
        return None
    candidates = []
    for name in os.listdir(runs_root):
        if not name.startswith("robot_nav_"):
            continue
        run_path = os.path.join(runs_root, name)
        if not os.path.isdir(run_path):
            continue
        ckpt_path = os.path.join(run_path, "best_model.pt")
        if not os.path.exists(ckpt_path):
            continue
        candidates.append((os.path.getmtime(ckpt_path), ckpt_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_init_checkpoint(init_ckpt_arg: str, runs_root: str) -> str | None:
    init_arg = init_ckpt_arg.strip()
    if not init_arg:
        return None
    if init_arg == "auto_latest_best":
        return _find_latest_best_checkpoint(runs_root)

    path = os.path.abspath(init_arg)
    if os.path.isdir(path):
        path = os.path.join(path, "best_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
    return path


def _extract_model_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


@dataclass
class LossWeights:
    term: float = 30.0
    stage: float = 1.0
    coll_soft: float = 40.0
    coll_hard: float = 100.0
    corridor: float = 50.0
    control: float = 0.05
    coll_margin: float = 0.1
    coll_beta: float = 10.0
    corridor_gap_crit: float = 0.08
    corridor_margin: float = 0.03
    corridor_beta: float = 12.0
    corridor_gap_beta: float = 20.0


@dataclass
class NoiseParams:
    sigma0: float
    tau: float


_PLT = None


def get_plt(show_plots: bool):
    global _PLT
    if _PLT is None:
        import matplotlib
        if not show_plots:
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        _PLT = plt
    return _PLT


def scenario_to_device(scenario, device: torch.device):
    return type(scenario)(
        start=scenario.start.to(device),
        goal=scenario.goal.to(device),
        centers=scenario.centers.to(device),
        radii=scenario.radii.to(device),
    )


def enforce_right_starts(scenario, right_frac: float, right_x_min: float, start_box: float, margin: float):
    """Overrides a fraction of starts to ensure they are on the right side and challenging."""
    B = scenario.start.shape[0]
    n_right = int(B * right_frac)
    if n_right <= 0:
        return scenario

    device = scenario.start.device
    starts = scenario.start.clone()
    centers = scenario.centers
    radii = scenario.radii

    for i in range(n_right):
        cand = None
        for _ in range(500):
            cx = torch.rand(1, device=device) * (start_box - right_x_min) + right_x_min
            cy = torch.rand(1, device=device) * (2 * start_box) - start_box
            c = torch.cat([cx, cy])

            # Check if it spawns inside an obstacle
            d = torch.norm(centers[i] - c.unsqueeze(0), dim=-1)
            if (d <= radii[i] + 0.1).any():
                continue

            # Verify the path is blocked (challenging)
            mask = direct_path_intersection_mask(
                c.unsqueeze(0),
                scenario.goal[i].unsqueeze(0),
                centers[i].unsqueeze(0),
                radii[i].unsqueeze(0),
                margin=margin
            )
            if mask[0]:
                cand = c
                break
        if cand is not None:
            starts[i] = cand

    scenario.start = starts
    return scenario


def sample_dataset(
        *,
        batch_size: int,
        seed: int,
        k: int,
        r_min: float,
        r_max: float,
        fixed_centers: torch.Tensor,
        start_box: float,
        challenge_frac: float,
        right_frac: float = 0.0,
        right_x_min: float = 0.5,
        right_margin: float = 0.05,
        device: torch.device,
):
    scenario_cpu = sample_scenarios(
        batch_size=batch_size,
        seed=seed,
        k=k,
        r_min=r_min,
        r_max=r_max,
        fixed_centers=fixed_centers,
        start_box=(-start_box, start_box),
    )
    scenario_cpu, ch_info = ensure_challenging_starts(
        scenario_cpu,
        start_box=(-start_box, start_box),
        min_fraction=challenge_frac,
        diverse_angles=True,
    )

    scenario = scenario_to_device(scenario_cpu, device)

    # Force 'right' challenge distribution for the local minimum problem
    if right_frac > 0.0:
        scenario = enforce_right_starts(scenario, right_frac, right_x_min, start_box, right_margin)

    return scenario, ch_info


def make_x0(scenario, device: torch.device) -> torch.Tensor:
    bsz = scenario.start.shape[0]
    return torch.cat(
        [scenario.start.to(device), torch.zeros(bsz, 2, device=device)],
        dim=-1,
    ).unsqueeze(1)


def make_decaying_noise(bsz: int, horizon: int, nx: int, params: NoiseParams, device: torch.device) -> torch.Tensor:
    t = torch.arange(horizon, device=device).view(1, horizon, 1)
    sigma_t = params.sigma0 * torch.exp(-t / max(params.tau, 1e-6))
    return torch.randn(bsz, horizon, nx, device=device) * sigma_t


def rollout_on_scenario(controller: PBController, plant_true: DoubleIntegratorTrue, scenario, horizon: int,
                        device: torch.device, noise: torch.Tensor | None):
    x0 = make_x0(scenario, device)

    def ctx_fn(x, t): return build_context(x, scenario)

    x_seq, u_seq, w_seq = rollout_bptt(
        controller=controller,
        plant_true=plant_true,
        x0=x0,
        horizon=horizon,
        context_fn=ctx_fn,
        w0=x0,
        process_noise_seq=noise,
    )
    return x_seq, u_seq, w_seq


def corridor_penalty(x_seq: torch.Tensor, scenario, weights: LossWeights) -> torch.Tensor:
    """Calculates a smooth repulsive bridge between impassable gaps."""
    pos = x_seq[..., :2]
    centers = scenario.centers
    radii = scenario.radii
    B, T, _ = pos.shape
    K = centers.shape[1]

    pen = torch.zeros(B, T, device=pos.device)
    if K < 2 or weights.corridor <= 0.0:
        return pen

    for i in range(K):
        for j in range(i + 1, K):
            c_i, c_j = centers[:, i], centers[:, j]
            r_i, r_j = radii[:, i], radii[:, j]

            # Gap between obstacles
            dist_obs = torch.norm(c_i - c_j, dim=-1)
            gap = dist_obs - r_i - r_j

            # Active if gap is less than the critical threshold
            gap_violation = torch.relu(weights.corridor_gap_crit - gap)
            gap_weight = torch.tanh(weights.corridor_gap_beta * gap_violation)

            d_i = torch.norm(pos - c_i.unsqueeze(1), dim=-1) - r_i.unsqueeze(1)
            d_j = torch.norm(pos - c_j.unsqueeze(1), dim=-1) - r_j.unsqueeze(1)

            # How deep into the corridor is the robot?
            excess_dist = (d_i + d_j) - gap.unsqueeze(1)

            # Softplus creates the smooth "wall" across the gap
            in_corridor = torch.nn.functional.softplus(
                weights.corridor_beta * (weights.corridor_margin - excess_dist)
            ) / weights.corridor_beta

            pen += gap_weight.unsqueeze(1) * in_corridor

    return pen


def compute_loss_per_sample(x_seq: torch.Tensor, u_seq: torch.Tensor, scenario, weights: LossWeights) -> torch.Tensor:
    pos_seq = x_seq[..., :2]
    dist_to_origin = torch.norm(pos_seq, dim=-1)

    term = dist_to_origin[:, -1] * weights.term
    stage = dist_to_origin.mean(dim=1) * weights.stage
    coll_soft = collision_penalty(
        x_seq, scenario, margin=weights.coll_margin, beta=weights.coll_beta
    ).mean(dim=(1, 2)) * weights.coll_soft
    coll_hard = collision_violation_penalty(x_seq, scenario).mean(dim=(1, 2)) * weights.coll_hard

    corridor = corridor_penalty(x_seq, scenario, weights).mean(dim=1) * weights.corridor
    control = (u_seq ** 2).mean(dim=(1, 2)) * weights.control

    return term + stage + coll_soft + coll_hard + corridor + control


def compute_loss(x_seq: torch.Tensor, u_seq: torch.Tensor, scenario, weights: LossWeights) -> tuple[torch.Tensor, dict]:
    total_per = compute_loss_per_sample(x_seq, u_seq, scenario, weights)
    total = total_per.mean()

    pos_seq = x_seq[..., :2]
    dist_to_origin = torch.norm(pos_seq, dim=-1)

    parts = {
        "loss_total": float(total.item()),
        "loss_term": float((dist_to_origin[:, -1].mean() * weights.term).item()),
        "loss_stage": float((dist_to_origin.mean() * weights.stage).item()),
        "loss_coll_soft": float((collision_penalty(x_seq, scenario, margin=weights.coll_margin,
                                                   beta=weights.coll_beta).mean() * weights.coll_soft).item()),
        "loss_coll_hard": float((collision_violation_penalty(x_seq, scenario).mean() * weights.coll_hard).item()),
        "loss_corridor": float((corridor_penalty(x_seq, scenario, weights).mean() * weights.corridor).item()),
        "loss_control": float(((u_seq ** 2).mean() * weights.control).item()),
    }
    return total, parts


@torch.no_grad()
def evaluate(controller: PBController, plant_true: DoubleIntegratorTrue, scenario, horizon: int, device: torch.device,
             weights: LossWeights):
    controller.eval()
    x_seq, u_seq, _ = rollout_on_scenario(
        controller=controller, plant_true=plant_true, scenario=scenario, horizon=horizon, device=device, noise=None
    )
    loss, parts = compute_loss(x_seq, u_seq, scenario, weights)
    min_dist = min_dist_to_edge(x_seq, scenario)

    collision_rate = float((min_dist.min(dim=1).values < 0).float().mean().item())

    # Calculate right-side specifically
    right_mask = scenario.start[:, 0] > 0.5
    if right_mask.sum() > 0:
        r_min_dist = min_dist[right_mask]
        right_coll = float((r_min_dist.min(dim=1).values < 0).float().mean().item())
    else:
        right_coll = collision_rate

    terminal_dist = float(torch.norm(x_seq[:, -1, :2], dim=-1).mean().item())

    parts.update({
        "loss": float(loss.item()),
        "collision_rate": collision_rate,
        "right_collision_rate": right_coll,
        "terminal_dist": terminal_dist,
    })
    return parts, x_seq, u_seq


def plot_loss_curves(train_hist, eval_hist, run_dir, show_plots: bool):
    plt = get_plt(show_plots)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([d["epoch"] for d in train_hist], [d["loss"] for d in train_hist], label="train", alpha=0.8)
    if eval_hist:
        ax.plot([d["epoch"] for d in eval_hist], [d["loss"] for d in eval_hist], label="val", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training/Validation Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"))
    if not show_plots:
        plt.close(fig)


def plot_trajectories(x_seq, scenario, run_dir, start_box: float, show_plots: bool, max_plots: int = 4):
    plt = get_plt(show_plots)
    num = min(max_plots, x_seq.shape[0])
    fig, axes = plt.subplots(1, num, figsize=(5 * num, 5), squeeze=False)
    for i in range(num):
        ax = axes[0, i]
        traj = x_seq[i, :, :2].detach().cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], color="C0")
        ax.scatter([scenario.start[i, 0].item()], [scenario.start[i, 1].item()], color="green", label="start")
        ax.scatter([0.0], [0.0], color="red", marker="*", s=100, label="goal")
        centers = scenario.centers[i].detach().cpu().numpy()
        radii = scenario.radii[i].detach().cpu().numpy()
        for k in range(centers.shape[0]):
            circ = plt.Circle((centers[k, 0], centers[k, 1]), radii[k], color="gray", alpha=0.3)
            ax.add_patch(circ)
        ax.set_xlim(-start_box, start_box)
        ax.set_ylim(-start_box, start_box)
        ax.set_aspect("equal", "box")
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "trajectories.png"))
    if not show_plots:
        plt.close(fig)


def plot_min_dist_hist(x_seq, scenario, run_dir, show_plots: bool):
    plt = get_plt(show_plots)
    min_dist = min_dist_to_edge(x_seq, scenario).detach().cpu().numpy().reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(min_dist, bins=30, density=True, alpha=0.7)
    ax.set_title("Min Distance to Obstacle Edge")
    ax.set_xlabel("distance")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "min_dist_hist.png"))
    if not show_plots:
        plt.close(fig)


def plot_m_outputs_over_time(u_seq, run_dir, show_plots: bool, num_samples: int = 6):
    plt = get_plt(show_plots)
    num = min(num_samples, u_seq.shape[0])
    t_steps = u_seq.shape[1]
    ts = torch.arange(t_steps).cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for i in range(num):
        ui = u_seq[i].detach().cpu().numpy()
        axes[0].plot(ts, ui[:, 0], alpha=0.8)
        axes[1].plot(ts, ui[:, 1], alpha=0.8)
    u_norm = torch.norm(u_seq.detach(), dim=-1).mean(dim=0).cpu().numpy()
    axes[2].plot(ts, u_norm, color="black", linewidth=2)
    axes[0].set_ylabel("u_x")
    axes[1].set_ylabel("u_y")
    axes[2].set_ylabel("mean ||u||")
    axes[2].set_xlabel("time step")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "m_outputs_over_time.png"))
    if not show_plots:
        plt.close(fig)


def _select_radius_case_indices(radii: torch.Tensor) -> List[int]:
    if radii.ndim != 2:
        raise ValueError(f"Expected radii shape (B,K), got {tuple(radii.shape)}")
    mean_r = radii.mean(dim=1)
    idx_small = int(torch.argmin(mean_r).item())
    idx_large = int(torch.argmax(mean_r).item())
    idx_med = int(torch.argsort(mean_r)[len(mean_r) // 2].item())
    idxs = [idx_small, idx_med, idx_large]
    unique = []
    for i in idxs:
        if i not in unique:
            unique.append(i)
    return unique


@torch.no_grad()
def plot_loss_heatmap_radius_levels(
        controller: PBController, plant_true: DoubleIntegratorTrue, ref_scenario, horizon: int, device: torch.device,
        weights: LossWeights, run_dir: str, show_plots: bool, start_box: float, heatmap_res: int, heatmap_batch: int,
):
    plt = get_plt(show_plots)
    res = int(heatmap_res)
    chunk = int(max(1, heatmap_batch))
    r = float(start_box)

    xs = torch.linspace(-r, r, res, device=device)
    ys = torch.linspace(-r, r, res, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    starts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    n_grid = starts.shape[0]

    case_indices = _select_radius_case_indices(ref_scenario.radii)[:3]
    if not case_indices: return
    labels = ["small", "median", "large"][: len(case_indices)]

    centers_base = ref_scenario.centers[0]
    grid_tensors = []
    radii_levels = []

    controller.eval()
    for idx in case_indices:
        radii_level = ref_scenario.radii[idx]
        losses_all, invalid_all = [], []

        for j0 in range(0, n_grid, chunk):
            j1 = min(j0 + chunk, n_grid)
            starts_chunk = starts[j0:j1]
            bsz = starts_chunk.shape[0]
            scenario_chunk = type(ref_scenario)(
                start=starts_chunk,
                goal=torch.zeros(bsz, 2, device=device),
                centers=centers_base.unsqueeze(0).expand(bsz, -1, -1),
                radii=radii_level.unsqueeze(0).expand(bsz, -1),
            )
            x_seq, u_seq, _ = rollout_on_scenario(controller, plant_true, scenario_chunk, horizon, device, None)
            losses = compute_loss_per_sample(x_seq, u_seq, scenario_chunk, weights)
            losses_all.append(losses)

            rel = starts_chunk.unsqueeze(1) - scenario_chunk.centers
            invalid = (torch.norm(rel, dim=-1) <= scenario_chunk.radii).any(dim=1)
            invalid_all.append(invalid)

        losses_full = torch.cat(losses_all, dim=0).clone()
        invalid_full = torch.cat(invalid_all, dim=0)
        losses_full[invalid_full] = float("nan")

        grid_tensors.append(losses_full.view(res, res).cpu())
        radii_levels.append(radii_level.detach().cpu())

    finite_vals = [g[torch.isfinite(g)] for g in grid_tensors if torch.isfinite(g).any()]
    if finite_vals:
        finite_all = torch.cat(finite_vals)
        vmin = float(torch.quantile(finite_all, 0.05).item())
        vmax = float(torch.quantile(finite_all, 0.95).item())
        if vmax <= vmin + 1e-9: vmax = vmin + 1e-6
    else:
        vmin, vmax = None, None

    n = len(grid_tensors)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    im = None
    centers_np = centers_base.detach().cpu().numpy()

    for j in range(n):
        ax = axes[0, j]
        im = ax.imshow(
            grid_tensors[j].numpy().T, origin="lower", extent=[-r, r, -r, r],
            aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        radii_np = radii_levels[j].numpy()
        for k in range(centers_np.shape[0]):
            ax.add_patch(
                plt.Circle((centers_np[k, 0], centers_np[k, 1]), radii_np[k], color="white", alpha=0.5, fill=True))
            ax.add_patch(
                plt.Circle((centers_np[k, 0], centers_np[k, 1]), radii_np[k], color="black", fill=False, linewidth=1.0))
        lbl = labels[j] if j < len(labels) else f"case {j + 1}"
        ax.set_title(f"{lbl} radii | mean r={float(radii_np.mean()):.2f}")
        ax.set_xlabel("x")
        if j == 0: ax.set_ylabel("y")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="cumulative loss")
    fig.suptitle("Loss Heatmap vs Start Position at 3 Radius Levels", y=0.98)
    fig.subplots_adjust(wspace=0.22, top=0.86, right=0.92)
    fig.savefig(os.path.join(run_dir, "loss_heatmap_radius_levels.png"))
    if not show_plots:
        plt.close(fig)


@torch.no_grad()
def plot_radius_comparison_challenging(
        controller: PBController, plant_true: DoubleIntegratorTrue, ref_scenario, horizon: int, device: torch.device,
        run_dir: str, show_plots: bool, start_box: float, num_starts: int, anchor_x: float, anchor_y: float,
        margin: float,
):
    plt = get_plt(show_plots)
    case_indices = _select_radius_case_indices(ref_scenario.radii)[:3]
    if not case_indices: return

    radius_cases = ref_scenario.radii[case_indices]
    centers = ref_scenario.centers[0]
    r_levels = radius_cases.shape[0]

    anchor = torch.tensor([anchor_x, anchor_y], dtype=centers.dtype, device=device)
    base_offsets = torch.tensor(
        [[0.0, 0.0], [0.0, 0.6], [0.0, -0.6], [-0.5, 0.35], [-0.5, -0.35], [0.4, 0.3], [0.4, -0.3], [-0.8, 0.0]],
        dtype=centers.dtype, device=device,
    )
    preset = (anchor.unsqueeze(0) + base_offsets).clamp(min=-start_box, max=start_box)
    large_r = radius_cases[-1]

    def _valid_and_challenging(starts: torch.Tensor):
        if starts.numel() == 0: return torch.zeros(0, dtype=torch.bool, device=device), torch.zeros(0, dtype=torch.bool,
                                                                                                    device=device)
        valid = (torch.norm(starts.unsqueeze(1) - centers.unsqueeze(0), dim=-1) > large_r.unsqueeze(0)).all(dim=1)
        challenging = direct_path_intersection_mask(
            starts=starts, goals=torch.zeros(starts.shape[0], 2, dtype=starts.dtype, device=device),
            centers=centers.unsqueeze(0).expand(starts.shape[0], -1, -1),
            radii=large_r.unsqueeze(0).expand(starts.shape[0], -1), margin=margin,
        )
        return valid, challenging

    valid_preset, chal_preset = _valid_and_challenging(preset)
    selected = [preset[i] for i in range(preset.shape[0]) if valid_preset[i] and chal_preset[i]][:num_starts]

    if len(selected) < num_starts:
        starts_all, goals_all = ref_scenario.start, ref_scenario.goal
        valid_all = (torch.norm(starts_all.unsqueeze(1) - centers.unsqueeze(0), dim=-1) > large_r.unsqueeze(0)).all(
            dim=1)
        chal_all = direct_path_intersection_mask(
            starts=starts_all, goals=goals_all, centers=ref_scenario.centers,
            radii=large_r.unsqueeze(0).expand(starts_all.shape[0], -1), margin=margin,
        )
        mask = valid_all & chal_all
        if mask.sum().item() == 0: mask = valid_all
        candidates = starts_all[mask]

        if candidates.shape[0] > 0:
            order = torch.argsort(torch.norm(candidates - anchor.unsqueeze(0), dim=1))
            for idx in order.tolist():
                if len(selected) >= num_starts: break
                cand = candidates[idx]
                if not any(torch.norm(cand - s).item() < 1e-5 for s in selected): selected.append(cand)

    if not selected: return

    starts = torch.stack(selected[:num_starts], dim=0)
    s = starts.shape[0]

    starts_batch = starts.unsqueeze(1).expand(s, r_levels, 2).reshape(s * r_levels, 2)
    radii_batch = radius_cases.unsqueeze(0).expand(s, r_levels, -1).reshape(s * r_levels, -1)
    centers_batch = centers.unsqueeze(0).expand(s * r_levels, -1, -1)
    goals_batch = torch.zeros(s * r_levels, 2, dtype=starts.dtype, device=device)

    scenario_cmp = type(ref_scenario)(start=starts_batch, goal=goals_batch, centers=centers_batch, radii=radii_batch)
    x_seq, _, _ = rollout_on_scenario(controller, plant_true, scenario_cmp, horizon, device, None)
    x_seq = x_seq.view(s, r_levels, x_seq.shape[1], x_seq.shape[2])

    labels = ["small", "median", "large"][:r_levels]
    colors, styles = ["C0", "C1", "C3"][:r_levels], ["-", "--", "-"][:r_levels]

    ncols = 2
    nrows = (s + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)
    centers_np, radius_cases_np = centers.detach().cpu().numpy(), radius_cases.detach().cpu().numpy()

    for i in range(s):
        ax = axes[i // ncols, i % ncols]
        for j in range(r_levels):
            traj = x_seq[i, j, :, :2].detach().cpu().numpy()
            lbl = labels[j] if j < len(labels) else f"level {j + 1}"
            ax.plot(traj[:, 0], traj[:, 1], color=colors[j], linestyle=styles[j], linewidth=2.0,
                    label=f"{lbl} (mean r={float(radius_cases_np[j].mean()):.2f})")
            for kk in range(centers_np.shape[0]):
                ax.add_patch(plt.Circle((centers_np[kk, 0], centers_np[kk, 1]), radius_cases_np[j, kk], color=colors[j],
                                        fill=False, alpha=0.35, linestyle=styles[j], linewidth=1.0))

        ax.scatter([starts[i, 0]], [starts[i, 1]], color="green", s=30, label="start")
        ax.scatter([0.0], [0.0], color="red", marker="*", s=80, label="goal")
        ax.set_title(f"Start {i + 1}: ({float(starts[i, 0]):.2f}, {float(starts[i, 1]):.2f})")
        ax.set_xlim(-start_box, start_box)
        ax.set_ylim(-start_box, start_box)
        ax.set_aspect("equal", "box")
        ax.grid(True)

    for i in range(s, nrows * ncols): axes[i // ncols, i % ncols].axis("off")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right")
    fig.suptitle("Challenging Radius Comparison (increasing radii)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "radius_comparison_challenging.png"), bbox_inches="tight")
    if not show_plots: plt.close(fig)

    with open(os.path.join(run_dir, "radius_comparison_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "case_indices": [int(i) for i in case_indices],
            "case_mean_radii": [float(v) for v in radius_cases.mean(dim=1).detach().cpu().tolist()],
            "anchor_start": [float(anchor[0].item()), float(anchor[1].item())],
            "selected_starts": [[float(x), float(y)] for x, y in starts.detach().cpu().tolist()],
        }, f, indent=2)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    show_plots = not args.no_show_plots
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(script_dir, "runs", "nav_experiment", f"robot_nav_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    weights = LossWeights(
        term=args.w_term, stage=args.w_stage, coll_soft=args.w_coll_soft,
        coll_hard=args.w_coll_hard, corridor=args.w_corridor, control=args.w_control,
        coll_margin=args.coll_margin, coll_beta=args.coll_beta,
        corridor_gap_crit=args.corridor_gap_crit, corridor_margin=args.corridor_margin,
        corridor_beta=args.corridor_beta, corridor_gap_beta=args.corridor_gap_beta,
    )
    noise_params = NoiseParams(sigma0=args.noise_sigma0, tau=args.noise_tau)

    plant_nom = DoubleIntegratorNominal(dt=0.05, pre_kp=1.0, pre_kd=1.5)
    plant_true = DoubleIntegratorTrue(dt=0.05, pre_kp=1.0, pre_kd=1.5)

    w_dim, u_dim, k, feat_dim = 4, 2, args.k_obstacles, 16
    z_dim = 2 + 2 * k + k + k

    fixed_centers = torch.tensor([[1.0, 0.0], [0.3, 0.8], [0.3, -0.8]], dtype=torch.float32)
    if k != 3: raise ValueError("robot_nav.py currently expects k_obstacles=3 because fixed centers are hard-coded.")

    mp = MpDeepSSM(w_dim, feat_dim, mode="loop", param="lru", n_layers=4, d_model=16, d_state=32, ff="GLU").to(device)
    mb = BoundedMLPOperator(w_dim=w_dim, z_dim=z_dim, r=u_dim, s=feat_dim, hidden_dim=64, bound_mode="softsign",
                            clamp_value=10.0).to(device)
    controller = PBController(plant=plant_nom, operator=FactorizedOperator(mp, mb), u_nominal=None, u_dim=u_dim,
                              detach_state=False).to(device)

    runs_root = os.path.join(script_dir, "runs", "nav_experiment")
    init_ckpt_path = resolve_init_checkpoint(args.init_ckpt, runs_root)
    if init_ckpt_path is not None:
        ckpt = torch.load(init_ckpt_path, map_location=device)
        controller.load_state_dict(_extract_model_state_dict(ckpt))
        print(f"Initialized controller from checkpoint: {init_ckpt_path}")
    else:
        print("Initialized controller from scratch.")

    optimizer = optim.Adam(controller.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    val_scenario, val_ch_info = sample_dataset(
        batch_size=args.val_batch, seed=args.seed + 999, k=k, r_min=args.r_min_end, r_max=args.r_max_end,
        fixed_centers=fixed_centers, start_box=args.start_box, challenge_frac=args.challenge_frac,
        right_frac=args.right_challenge_frac_val, right_x_min=args.right_x_min,
        right_margin=args.right_challenge_margin, device=device
    )

    print(f"Starting Training on {device} without u_nominal...")

    train_hist, eval_hist = [], []
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")

    init_metrics, _, _ = evaluate(controller, plant_true, val_scenario, args.horizon, device, weights)
    init_metrics["epoch"] = 0
    eval_hist.append(init_metrics)

    best_eval_loss = float(init_metrics["loss"])
    best_eval_collision = float(
        init_metrics["right_collision_rate"] if args.best_ckpt_metric == "right_collision_then_loss" else init_metrics[
            "collision_rate"])
    best_epoch = 0

    torch.save({
        "epoch": 0, "model_state_dict": controller.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "best_eval_loss": best_eval_loss, "best_eval_collision": best_eval_collision, "args": vars(args),
    }, best_ckpt_path)

    print(
        f"Epoch 000/{args.epochs} | Val {init_metrics['loss']:.2f} | Val Right-Crash {init_metrics['right_collision_rate'] * 100:.1f}% | (pre-train baseline)")

    for epoch in range(1, args.epochs + 1):
        controller.train()
        optimizer.zero_grad()

        progress = min(1.0, float(epoch) / float(max(args.curriculum_epochs, 1)))
        r_min = args.r_min_start + (args.r_min_end - args.r_min_start) * progress
        r_max = args.r_max_start + (args.r_max_end - args.r_max_start) * progress

        train_scenario, train_ch_info = sample_dataset(
            batch_size=args.batch, seed=args.seed + epoch, k=k, r_min=r_min, r_max=r_max,
            fixed_centers=fixed_centers, start_box=args.start_box, challenge_frac=args.challenge_frac,
            right_frac=args.right_challenge_frac_train, right_x_min=args.right_x_min,
            right_margin=args.right_challenge_margin, device=device
        )

        noise = make_decaying_noise(args.batch, args.horizon, w_dim, noise_params, device)
        x_seq, u_seq, _ = rollout_on_scenario(controller, plant_true, train_scenario, args.horizon, device, noise)
        loss, train_parts = compute_loss(x_seq, u_seq, train_scenario, weights)

        loss.backward()
        if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        scheduler.step()

        train_hist.append({"epoch": epoch, "loss": float(loss.item())})

        if epoch % args.eval_every == 0 or epoch == 1:
            metrics, _, _ = evaluate(controller, plant_true, val_scenario, args.horizon, device, weights)
            metrics["epoch"] = epoch
            eval_hist.append(metrics)

            if args.best_ckpt_metric == "loss":
                better_ckpt = metrics["loss"] < best_eval_loss
            elif args.best_ckpt_metric == "right_collision_then_loss":
                coll = float(metrics["right_collision_rate"])
                tol = float(args.best_ckpt_collision_tol)
                better_ckpt = (coll < (best_eval_collision - tol) or (
                            abs(coll - best_eval_collision) <= tol and metrics["loss"] < best_eval_loss))
            else:
                coll = float(metrics["collision_rate"])
                tol = float(args.best_ckpt_collision_tol)
                better_ckpt = (coll < (best_eval_collision - tol) or (
                            abs(coll - best_eval_collision) <= tol and metrics["loss"] < best_eval_loss))

            if better_ckpt:
                best_eval_loss = float(metrics["loss"])
                best_eval_collision = coll
                best_epoch = epoch
                torch.save({
                    "epoch": epoch, "model_state_dict": controller.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eval_loss": best_eval_loss, "best_eval_collision": best_eval_collision, "args": vars(args),
                }, best_ckpt_path)

            print(
                f"Epoch {epoch:03d}/{args.epochs} | Train {train_parts['loss_total']:.2f} | Val {metrics['loss']:.2f} | "
                f"Val Crash(R) {metrics['right_collision_rate'] * 100:.1f}% | r_max {r_max:.2f} | LR {scheduler.get_last_lr()[0]:.1e}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        controller.load_state_dict(_extract_model_state_dict(ckpt))

    final_metrics, x_val, u_val = evaluate(controller, plant_true, val_scenario, args.horizon, device, weights)
    final_metrics.update({
        "best_epoch": int(best_epoch), "best_eval_loss": float(best_eval_loss),
        "best_eval_collision": float(best_eval_collision),
        "best_ckpt_metric": args.best_ckpt_metric, "val_challenging_fraction": float(val_ch_info["after_fraction"]),
        "train_epochs": int(args.epochs), "k_obstacles": int(k), "init_ckpt_path": init_ckpt_path or "",
    })

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    with open(os.path.join(run_dir, "eval_history.json"), "w", encoding="utf-8") as f:
        json.dump(eval_hist, f, indent=2)
    with open(os.path.join(run_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_hist, f, indent=2)

    plot_loss_curves(train_hist, eval_hist, run_dir, show_plots=show_plots)
    plot_trajectories(x_val, val_scenario, run_dir, start_box=args.start_box, show_plots=show_plots, max_plots=4)
    plot_min_dist_hist(x_val, val_scenario, run_dir, show_plots=show_plots)
    plot_m_outputs_over_time(u_val, run_dir, show_plots=show_plots)
    plot_loss_heatmap_radius_levels(controller, plant_true, val_scenario, args.horizon, device, weights, run_dir,
                                    show_plots, args.start_box, args.heatmap_res, args.heatmap_batch)
    plot_radius_comparison_challenging(controller, plant_true, val_scenario, args.horizon, device, run_dir, show_plots,
                                       args.start_box, args.radius_cmp_num_starts, args.radius_cmp_anchor_x,
                                       args.radius_cmp_anchor_y, args.radius_cmp_margin)

    torch.save(controller.state_dict(), os.path.join(run_dir, "pb_model_final.pt"))
    print(f"Training complete. Artifacts saved to {run_dir}")

    if show_plots:
        plt = get_plt(True)
        plt.show()


if __name__ == "__main__":
    main()