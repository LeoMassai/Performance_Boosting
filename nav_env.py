"""Scenario sampling and context utilities for 2D navigation with circular obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class NavScenario:
    start: torch.Tensor   # (B, 2)
    goal: torch.Tensor    # (B, 2)
    centers: torch.Tensor # (B, K, 2)
    radii: torch.Tensor   # (B, K)


def sample_scenarios(
    batch_size: int,
    seed: int,
    k: int = 3,
    r_min: float = 0.2,
    r_max: float = 0.6,
    start_box: Tuple[float, float] = (-3.0, 3.0),
    center_box: Tuple[float, float] = (-1.5, 1.5),
    start_clearance: float = 0.4,
    goal_clearance: float = 0.4,
    obstacle_clearance: float = 0.2,
    fixed_centers: Optional[torch.Tensor] = None,
    max_attempts: int = 200,
) -> NavScenario:
    rng = np.random.RandomState(seed)

    starts = np.zeros((batch_size, 2), dtype=np.float32)
    goals = np.zeros((batch_size, 2), dtype=np.float32)
    centers = np.zeros((batch_size, k, 2), dtype=np.float32)
    radii = np.zeros((batch_size, k), dtype=np.float32)

    for b in range(batch_size):
        for _ in range(max_attempts):
            start = _sample_start_box(rng, start_box)
            goal = np.zeros(2, dtype=np.float32)

            obs_centers = []
            obs_radii = []
            ok = True

            if fixed_centers is not None:
                fixed = fixed_centers.detach().cpu().numpy()
                if fixed.shape != (k, 2):
                    raise ValueError(f"fixed_centers must have shape (K,2), got {fixed.shape}")
                obs_centers = list(fixed)
                for _ in range(k):
                    r = rng.uniform(r_min, r_max)
                    obs_radii.append(r)
            else:
                for _ in range(k):
                    placed = False
                    for _ in range(max_attempts):
                        r = rng.uniform(r_min, r_max)
                        c = rng.uniform(center_box[0], center_box[1], size=2).astype(np.float32)

                        if np.linalg.norm(c - start) <= r + start_clearance:
                            continue
                        if np.linalg.norm(c - goal) <= r + goal_clearance:
                            continue
                        if obs_centers:
                            prev_c = np.stack(obs_centers, axis=0)
                            prev_r = np.array(obs_radii)
                            d = np.linalg.norm(prev_c - c[None, :], axis=1)
                            if np.any(d <= (prev_r + r + obstacle_clearance)):
                                continue
                        obs_centers.append(c)
                        obs_radii.append(r)
                        placed = True
                        break
                    if not placed:
                        ok = False
                        break

                if not ok:
                    continue

            obs_centers_arr = np.stack(obs_centers, axis=0)
            obs_radii_arr = np.array(obs_radii)

            # ensure start/goal are not inside any obstacle
            dist_start = np.linalg.norm(obs_centers_arr - start[None, :], axis=1)
            dist_goal = np.linalg.norm(obs_centers_arr - goal[None, :], axis=1)
            if np.any(dist_start <= (obs_radii_arr + start_clearance)):
                ok = False
            if np.any(dist_goal <= (obs_radii_arr + goal_clearance)):
                ok = False

            if not ok:
                continue

            starts[b] = start
            goals[b] = goal
            centers[b] = obs_centers_arr
            radii[b] = obs_radii_arr
            break
        else:
            raise RuntimeError("Failed to sample a valid scenario. Try relaxing constraints.")

    return NavScenario(
        start=torch.from_numpy(starts),
        goal=torch.from_numpy(goals),
        centers=torch.from_numpy(centers),
        radii=torch.from_numpy(radii),
    )


def direct_path_intersection_mask(
    starts: torch.Tensor,
    goals: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Return a boolean mask (B,) indicating whether the straight segment start->goal
    intersects any obstacle enlarged by `margin`.
    """
    starts = starts.float()
    goals = goals.float()
    if centers.dim() == 2:
        centers = centers.unsqueeze(0).expand(starts.shape[0], -1, -1)
    if radii.dim() == 1:
        radii = radii.unsqueeze(0).expand(starts.shape[0], -1)

    seg = goals - starts  # (B,2)
    seg2 = (seg * seg).sum(dim=-1, keepdim=True).clamp_min(1e-9)  # (B,1)
    rel = centers - starts.unsqueeze(1)  # (B,K,2)
    t = (rel * seg.unsqueeze(1)).sum(dim=-1) / seg2  # (B,K)
    t = t.clamp(0.0, 1.0)
    closest = starts.unsqueeze(1) + t.unsqueeze(-1) * seg.unsqueeze(1)  # (B,K,2)
    dist = torch.norm(centers - closest, dim=-1)  # (B,K)

    return (dist <= (radii + margin)).any(dim=1)


def ensure_challenging_starts(
    scenario: NavScenario,
    start_box: Tuple[float, float],
    min_fraction: float = 0.35,
    margin: float = 0.05,
    radius_quantile: float = 0.9,
    start_clearance: float = 0.05,
    max_attempts: int = 300,
    cone_fraction: float = 0.75,
    cone_angle_spread: float = 1.2,
    cone_radial_extension: float = 1.8,
    diverse_angles: bool = False,
    num_angle_bins: int = 16,
) -> Tuple[NavScenario, dict]:
    """
    Ensure a minimum fraction of starts have straight-line paths to goal intersecting
    enlarged obstacles computed from high-quantile radii.

    Replacement starts are sampled with a mixed strategy:
      - cone sampling around a reference obstacle direction (default),
      - uniform fallback in the whole start box.

    If `diverse_angles=True`, replacements are angle-balanced around the goal
    (uniform challenging starts across angular bins), which avoids one-sided
    datasets and increases blocked straight-line coverage from many directions.
    """
    if min_fraction <= 0:
        info = {
            "before_fraction": 0.0,
            "after_fraction": 0.0,
            "target_fraction": float(min_fraction),
            "replaced": 0,
            "cone_replaced": 0,
        }
        return scenario, info

    starts = scenario.start.clone()
    goals = scenario.goal
    centers = scenario.centers
    radii = scenario.radii
    bsz = starts.shape[0]

    q = float(min(max(radius_quantile, 0.0), 1.0))
    radii_ref = torch.quantile(radii, q=q, dim=0)  # (K,)
    radii_ref_batch = radii_ref.unsqueeze(0).expand(bsz, -1)

    mask_before = direct_path_intersection_mask(
        starts=starts,
        goals=goals,
        centers=centers,
        radii=radii_ref_batch,
        margin=margin,
    )
    before_count = int(mask_before.sum().item())
    target_count = int(np.ceil(min_fraction * bsz))
    if before_count >= target_count:
        info = {
            "before_fraction": before_count / float(bsz),
            "after_fraction": before_count / float(bsz),
            "target_fraction": float(min_fraction),
            "replaced": 0,
            "cone_replaced": 0,
        }
        return scenario, info

    need = target_count - before_count
    non_challenging_idx = torch.nonzero(~mask_before, as_tuple=False).squeeze(-1)
    replaced = 0
    cone_replaced = 0

    cone_fraction = float(min(max(cone_fraction, 0.0), 1.0))
    cone_angle_spread = float(max(cone_angle_spread, 1e-3))
    cone_radial_extension = float(max(cone_radial_extension, 0.05))
    box_min = float(start_box[0])
    box_max = float(start_box[1])
    abs_box = max(abs(box_min), abs(box_max))

    use_diverse_angles = bool(diverse_angles and num_angle_bins >= 2)
    num_bins = int(max(2, num_angle_bins)) if use_diverse_angles else 0
    if use_diverse_angles:
        bins_before = angle_bins_for_starts(
            starts=starts,
            goals=goals,
            num_bins=num_bins,
        )
        bin_counts = torch.bincount(bins_before[mask_before], minlength=num_bins).to(starts.device)
    else:
        bin_counts = None

    for idx in non_challenging_idx.tolist():
        if replaced >= need:
            break

        c_i = centers[idx]
        r_i = radii[idx]
        replaced_this_sample = False

        target_bin = None
        if use_diverse_angles:
            target_bin = int(torch.argmin(bin_counts).item())

        for attempt in range(max_attempts):
            use_cone = False
            if not use_diverse_angles:
                use_cone = bool(torch.rand((), device=starts.device).item() < cone_fraction)

            if use_cone:
                # Reference the largest obstacle in this scenario to force non-trivial detours.
                k_ref = int(torch.argmax(r_i).item())
                c_ref = c_i[k_ref]
                r_ref = r_i[k_ref]

                phi_center = torch.atan2(c_ref[1], c_ref[0])
                phi = torch.empty((), dtype=starts.dtype, device=starts.device).uniform_(
                    float(phi_center.item()) - 0.5 * cone_angle_spread,
                    float(phi_center.item()) + 0.5 * cone_angle_spread,
                )

                rho_min = float((torch.norm(c_ref) + r_ref + start_clearance).item())
                rho_max = min(abs_box, rho_min + cone_radial_extension)
                if rho_max <= rho_min + 1e-6:
                    continue

                rho = torch.empty((), dtype=starts.dtype, device=starts.device).uniform_(rho_min, rho_max)
                cand = torch.stack([rho * torch.cos(phi), rho * torch.sin(phi)], dim=0)
                cand = cand.clamp(min=box_min, max=box_max)
            else:
                cand = torch.empty(2, dtype=starts.dtype, device=starts.device).uniform_(box_min, box_max)

            # Keep candidate outside actual obstacle bodies for that sample.
            if torch.any(torch.norm(c_i - cand.unsqueeze(0), dim=-1) <= (r_i + start_clearance)):
                continue

            is_challenging = direct_path_intersection_mask(
                starts=cand.unsqueeze(0),
                goals=goals[idx].unsqueeze(0),
                centers=c_i.unsqueeze(0),
                radii=radii_ref.unsqueeze(0),
                margin=margin,
            )[0]
            if not bool(is_challenging.item()):
                continue

            if use_diverse_angles:
                cand_bin = angle_bin_single(
                    start=cand,
                    goal=goals[idx],
                    num_bins=num_bins,
                )
                enforce_target_bin = attempt < (max_attempts // 2)
                if enforce_target_bin and cand_bin != target_bin:
                    continue
                bin_counts[cand_bin] += 1

            starts[idx] = cand
            replaced += 1
            cone_replaced += int(use_cone)
            replaced_this_sample = True
            break

        if not replaced_this_sample:
            continue

    scenario = NavScenario(
        start=starts,
        goal=goals,
        centers=centers,
        radii=radii,
    )

    mask_after = direct_path_intersection_mask(
        starts=scenario.start,
        goals=scenario.goal,
        centers=scenario.centers,
        radii=radii_ref_batch,
        margin=margin,
    )
    after_count = int(mask_after.sum().item())
    info = {
        "before_fraction": before_count / float(bsz),
        "after_fraction": after_count / float(bsz),
        "target_fraction": float(min_fraction),
        "replaced": int(replaced),
        "cone_replaced": int(cone_replaced),
        "diverse_angles": bool(use_diverse_angles),
        "num_angle_bins": int(num_bins if use_diverse_angles else 0),
    }
    if use_diverse_angles:
        bins_after = angle_bins_for_starts(
            starts=scenario.start,
            goals=scenario.goal,
            num_bins=num_bins,
        )
        info["angle_bin_counts_all"] = [int(v) for v in torch.bincount(bins_after, minlength=num_bins).cpu().tolist()]
        info["angle_bin_counts_challenging"] = [
            int(v) for v in torch.bincount(bins_after[mask_after], minlength=num_bins).cpu().tolist()
        ]
    return scenario, info


def angle_bins_for_starts(starts: torch.Tensor, goals: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Map start->goal directions to integer bins in [0, num_bins)."""
    if starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError(f"starts must have shape (B,2), got {tuple(starts.shape)}")
    if goals.ndim != 2 or goals.shape[1] != 2:
        raise ValueError(f"goals must have shape (B,2), got {tuple(goals.shape)}")
    if starts.shape[0] != goals.shape[0]:
        raise ValueError(f"starts/goals batch mismatch: {starts.shape[0]} vs {goals.shape[0]}")
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")

    vec = starts - goals
    theta = torch.atan2(vec[:, 1], vec[:, 0])
    frac = (theta + torch.pi) / (2.0 * torch.pi)
    bins = torch.floor(frac * float(num_bins)).long() % int(num_bins)
    return bins


def angle_bin_single(start: torch.Tensor, goal: torch.Tensor, num_bins: int) -> int:
    """Single-sample helper returning a python int bin index."""
    return int(angle_bins_for_starts(start.view(1, 2), goal.view(1, 2), num_bins=num_bins).item())



def _sample_start_box(
    rng: np.random.RandomState,
    box: Tuple[float, float],
) -> np.ndarray:
    x = rng.uniform(box[0], box[1])
    y = rng.uniform(box[0], box[1])
    return np.array([x, y], dtype=np.float32)


def build_context(x: torch.Tensor, scenario: NavScenario) -> torch.Tensor:
    """
    Build context z_t from state and scenario.
    Returns (B, T, ctx_dim) where ctx = [goal_dir, rel_pos, dist_to_edge, radii].
    """
    x = _as_bt(x)
    pos = x[:, :, :2]
    _, T, _ = pos.shape

    centers = scenario.centers.to(x.device).unsqueeze(1)  # (B,1,K,2)
    radii = scenario.radii.to(x.device).unsqueeze(1)      # (B,1,K)
    goal = scenario.goal.to(x.device).unsqueeze(1)        # (B,1,2)

    rel = centers - pos.unsqueeze(2)                      # (B,1,K,2)
    dist = torch.norm(rel, dim=-1)                        # (B,1,K)
    dist_to_edge = dist - radii

    goal_dir = goal - pos                                 # (B,1,2)
    radii_t = radii.expand(-1, T, -1)                    # (B,T,K)
    ctx = torch.cat(
        [
            goal_dir,
            rel.reshape(rel.shape[0], rel.shape[1], -1),
            dist_to_edge,
            radii_t,
        ],
        dim=-1,
    )
    return ctx



def obstacle_edge_distances(x: torch.Tensor, scenario: NavScenario, robot_radius: float = 0.0) -> torch.Tensor:
    """Return distance to each obstacle edge, shape (B, T, K)."""
    x = _as_bt(x)
    pos = x[:, :, :2]
    centers = scenario.centers.to(x.device).unsqueeze(1)  # (B,1,K,2)
    radii = scenario.radii.to(x.device).unsqueeze(1) + float(robot_radius)  # (B,1,K)

    rel = pos.unsqueeze(2) - centers
    dist = torch.norm(rel, dim=-1)
    return dist - radii

def collision_penalty(
    x: torch.Tensor,
    scenario: NavScenario,
    margin: float = 0.1,
    beta: float = 20.0,
) -> torch.Tensor:
    dist_to_edge = obstacle_edge_distances(x, scenario, robot_radius=0.0)

    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")

    # Smooth hinge barrier with adjustable sharpness.
    # Larger beta makes penalty decay faster when far from obstacles.
    pen = F.softplus(beta * (margin - dist_to_edge)) / beta
    return pen.sum(dim=-1, keepdim=True)  # (B, T, 1)


def collision_violation_penalty(x: torch.Tensor, scenario: NavScenario) -> torch.Tensor:
    """
    Harder penalty for actual obstacle penetration.
    Returns (B, T, 1).
    """
    dist_to_edge = obstacle_edge_distances(x, scenario, robot_radius=0.0)
    pen = torch.relu(-dist_to_edge) ** 2
    return pen.sum(dim=-1, keepdim=True)


def min_dist_to_edge(x: torch.Tensor, scenario: NavScenario) -> torch.Tensor:
    """Return min distance to any obstacle edge, shape (B, T)."""
    dist_to_edge = obstacle_edge_distances(x, scenario, robot_radius=0.0)
    min_dist = dist_to_edge.min(dim=-1).values
    return min_dist


def _as_bt(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected (B,N) or (B,T,N), got {tuple(x.shape)}")
