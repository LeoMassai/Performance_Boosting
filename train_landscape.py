"""
PB Navigation: The Cost Landscape & Turbulence Experiment.
Demonstrates Factorized PB Control WITHOUT a baseline (u_nominal=None).

1. M_b(w, z) acts as the Navigator, using context (z) to route around Gaussian Cost Hills.
2. M_p(w) acts as the Stabilizer, rejecting unmodeled crosswinds and nonlinear drag.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Your Custom Modules ---
from bounded_mlp_operator import BoundedMLPOperator
from ssm_operators import MpDeepSSM
from pb_controller import PBController, FactorizedOperator, as_bt
from rollout_bptt import rollout_bptt
from nav_plants import DoubleIntegratorNominal
from nav_env import sample_scenarios, ensure_challenging_starts, build_context


# ---------------------------------------------------------------------------
# 1. THE TRUE PLANT (UNMODELED DISTURBANCES)
# ---------------------------------------------------------------------------
class NoisyDoubleIntegratorTrue:
    """
    The True Plant adds nonlinear drag and a constant crosswind.
    Because the Nominal Plant doesn't know about this, w_t != 0.
    This gives the DeepSSM operator M_p(w) its primary job: disturbance rejection!
    """

    def __init__(self, dt: float = 0.05, pre_kp: float = 1.0, pre_kd: float = 1.5, wind_x: float = 0.3,
                 wind_y: float = 0.2):
        self.dt = dt
        self.pre_kp = pre_kp
        self.pre_kd = pre_kd
        self.wind = torch.tensor([wind_x, wind_y])

    def forward(self, x: torch.Tensor, u: torch.Tensor, t=None) -> torch.Tensor:
        x = as_bt(x)
        u = as_bt(u)
        pos, vel = x[..., :2], x[..., 2:]

        # Unmodeled Nonlinear Drag (quadratic)
        drag = 0.05 * vel * torch.norm(vel, dim=-1, keepdim=True)
        # Unmodeled Crosswind
        wind_tensor = self.wind.to(x.device).unsqueeze(0).unsqueeze(0)

        acc = (-self.pre_kp * pos) - (self.pre_kd * vel) + u - drag + wind_tensor

        pos_next = pos + self.dt * vel
        vel_next = vel + self.dt * acc
        return torch.cat([pos_next, vel_next], dim=-1)


# ---------------------------------------------------------------------------
# 2. THE SMOOTH COST LANDSCAPE PENALTY
# ---------------------------------------------------------------------------
def gaussian_hill_penalty(x_seq: torch.Tensor, scenario, height: float = 50.0) -> torch.Tensor:
    """
    Replaces the hard collision penalty.
    Creates smooth, differentiable mountains where the obstacles are.
    """
    pos = x_seq[..., :2]
    centers = scenario.centers.unsqueeze(1)  # (B, 1, K, 2)
    radii = scenario.radii.unsqueeze(1)  # (B, 1, K)

    # Squared distance from robot to each hill peak
    dist_sq = torch.sum((pos.unsqueeze(2) - centers) ** 2, dim=-1)  # (B, T, K)

    # Gaussian bell curve. sigma controls the width of the hill based on the radius
    sigma_sq = (radii / 1.5) ** 2 + 1e-6
    hills = height * torch.exp(-dist_sq / (2 * sigma_sq))

    # Sum the penalties from all K hills -> Shape (B, T)
    return hills.sum(dim=-1)


# ---------------------------------------------------------------------------
# 3. LOSS FUNCTION
# ---------------------------------------------------------------------------
def compute_loss(x_seq, u_seq, scenario):
    pos_seq = x_seq[..., :2]
    vel_seq = x_seq[..., 2:]

    dist_to_origin = torch.norm(pos_seq, dim=-1)

    # 1. Target Tracking (Reach origin and stop)
    term_pos = dist_to_origin[:, -1].mean() * 30.0
    term_vel = torch.norm(vel_seq[:, -1], dim=-1).mean() * 10.0
    stage = dist_to_origin.mean() * 2.0

    # 2. The Cost Landscape
    # BPTT naturally looks ahead and routes around these hills to minimize this integral!
    hills = gaussian_hill_penalty(x_seq, scenario, height=80.0).mean() * 1.0

    # 3. Control Effort & Smoothness (Crucial for Neural Controllers)
    control = (u_seq ** 2).mean() * 0.05
    du = ((u_seq[:, 1:] - u_seq[:, :-1]) ** 2).mean() * 0.5

    total_loss = term_pos + term_vel + stage + hills + control + du

    parts = {
        "loss": total_loss.item(),
        "term_dist": dist_to_origin[:, -1].mean().item(),
        "hill_cost": hills.item()
    }
    return total_loss, parts


# ---------------------------------------------------------------------------
# 4. PLOTTING (BEAUTIFUL MAGMA CONTOURS)
# ---------------------------------------------------------------------------
_PLT = None


def get_plt(show: bool):
    global _PLT
    if _PLT is None:
        import matplotlib
        if not show: matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        _PLT = plt
    return _PLT


def plot_landscape_and_trajectories(x_seq, scenario, run_dir, start_box=3.0, show_plots=False):
    plt = get_plt(show_plots)
    num = min(4, x_seq.shape[0])
    fig, axes = plt.subplots(1, num, figsize=(6 * num, 6), squeeze=False)

    res = 120
    r = float(start_box)
    xs = torch.linspace(-r, r, res, device=x_seq.device)
    ys = torch.linspace(-r, r, res, device=x_seq.device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    grid_pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (N, 2)

    for i in range(num):
        ax = axes[0, i]
        c_i = scenario.centers[i].unsqueeze(0)  # (1, K, 2)
        r_i = scenario.radii[i].unsqueeze(0)  # (1, K)

        # Calculate hill costs for the background heatmap
        dist_sq = torch.sum((grid_pts.unsqueeze(1) - c_i) ** 2, dim=-1)
        sigma_sq = (r_i / 1.5) ** 2 + 1e-6
        hills = 80.0 * torch.exp(-dist_sq / (2 * sigma_sq))
        Z = hills.sum(dim=-1).view(res, res).cpu().numpy()

        # Plot Heatmap
        im = ax.imshow(Z.T, origin="lower", extent=[-r, r, -r, r], cmap="magma", alpha=0.9)

        # Plot Trajectory over it
        traj = x_seq[i, :, :2].detach().cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], color="cyan", linewidth=3.0, label="Trajectory")
        ax.scatter([traj[0, 0]], [traj[0, 1]], color="lime", s=80, edgecolors='black', label="Start", zorder=5)
        ax.scatter([0.0], [0.0], color="white", marker="*", s=200, edgecolors='black', label="Goal", zorder=5)

        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Cost Landscape - Route {i + 1}")
        if i == 0:
            ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "landscape_trajectories.png"))
    if not show_plots: plt.close(fig)


@torch.no_grad()
def generate_eval_plots(controller, plant_true, run_dir, start_box=3.0, device="cpu"):
    """Generates the Heatmap and the Hill Size Comparison plots."""
    print("Generating evaluation plots...")
    plt = get_plt(show=False)

    # -----------------------------------------------------------------------
    # PLOT 1: LOSS HEATMAP (Dependence on Starting Condition)
    # -----------------------------------------------------------------------
    res = 80  # Resolution of the heatmap grid (80x80 = 6400 start points)
    r = float(start_box)
    xs = torch.linspace(-r, r, res, device=device)
    ys = torch.linspace(-r, r, res, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    starts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (6400, 2)
    B = starts.shape[0]

    # Standard scenario to test against
    fixed_centers = torch.tensor([[1.0, 0.0], [0.3, 0.8], [0.3, -0.8]], device=device)
    radii = torch.tensor([0.4, 0.4, 0.4], device=device)

    scen = type("Scenario", (), {})()
    scen.start = starts
    scen.goal = torch.zeros(B, 2, device=device)
    scen.centers = fixed_centers.unsqueeze(0).expand(B, -1, -1)
    scen.radii = radii.unsqueeze(0).expand(B, -1)

    x0 = torch.cat([starts, torch.zeros(B, 2, device=device)], dim=-1).unsqueeze(1)

    def ctx_fn(x, t):
        return build_context(x, scen)

    # Rollout in chunks to prevent GPU Out-of-Memory
    chunk_size = 2000
    losses = []
    controller.eval()

    for i in range(0, B, chunk_size):
        end = min(i + chunk_size, B)
        x0_chunk = x0[i:end]
        scen_chunk = type("Scenario", (), {})()
        scen_chunk.centers = scen.centers[i:end]
        scen_chunk.radii = scen.radii[i:end]
        scen_chunk.goal = scen.goal[i:end]

        def ctx_chunk(x, t): return build_context(x, scen_chunk)

        x_seq, u_seq, _ = rollout_bptt(controller, plant_true, x0_chunk, horizon=80, context_fn=ctx_chunk, w0=x0_chunk)
        loss_val, _ = compute_loss(x_seq, u_seq, scen_chunk)
        # We want per-sample loss, so we recompute just the term + stage + hills per sample
        pos = x_seq[..., :2]
        dist = torch.norm(pos, dim=-1)
        term = dist[:, -1] * 30.0
        stage = dist.mean(dim=-1) * 2.0
        hills = gaussian_hill_penalty(x_seq, scen_chunk, height=80.0).mean(dim=-1) * 1.0
        sample_loss = term + stage + hills
        losses.append(sample_loss.cpu())

    loss_grid = torch.cat(losses).view(res, res).numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(loss_grid.T, origin="lower", extent=[-r, r, -r, r], cmap="viridis",
                   vmax=np.percentile(loss_grid, 95))

    # Overlay the hills
    c_np = fixed_centers.cpu().numpy()
    r_np = radii.cpu().numpy()
    for k in range(c_np.shape[0]):
        ax.add_patch(
            plt.Circle((c_np[k, 0], c_np[k, 1]), r_np[k], color="white", fill=False, linestyle="--", linewidth=2.0))

    ax.scatter([0.0], [0.0], color="red", marker="*", s=200, label="Goal")
    ax.set_title("Total Loss Heatmap based on Start Position")
    fig.colorbar(im, ax=ax, label="Cumulative Loss")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_heatmap.png"))
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PLOT 2: HILL SIZE COMPARISON (Proving z_t contextual adaptation)
    # -----------------------------------------------------------------------
    # Pick 4 challenging start positions
    test_starts = torch.tensor([[-2.5, 0.0], [-2.0, 2.0], [-2.0, -2.0], [0.0, 2.5]], device=device)
    num_starts = test_starts.shape[0]

    r_levels = [0.2, 0.5, 0.8]
    colors = ["cyan", "orange", "magenta"]
    labels = ["Small Hills (r=0.2)", "Med Hills (r=0.5)", "Large Hills (r=0.8)"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(num_starts):
        ax = axes[i]
        start_pt = test_starts[i].unsqueeze(0)  # (1, 2)
        x0_single = torch.cat([start_pt, torch.zeros(1, 2, device=device)], dim=-1).unsqueeze(1)

        ax.scatter([start_pt[0, 0].item()], [start_pt[0, 1].item()], color="lime", s=100, edgecolors="black", zorder=5,
                   label="Start")
        ax.scatter([0.0], [0.0], color="red", marker="*", s=200, edgecolors="black", zorder=5, label="Goal")

        for j, r_val in enumerate(r_levels):
            scen_j = type("Scenario", (), {})()
            scen_j.start = start_pt
            scen_j.goal = torch.zeros(1, 2, device=device)
            scen_j.centers = fixed_centers.unsqueeze(0)
            scen_j.radii = torch.full((1, 3), r_val, device=device)

            def ctx_j(x, t):
                return build_context(x, scen_j)

            x_seq, _, _ = rollout_bptt(controller, plant_true, x0_single, horizon=80, context_fn=ctx_j, w0=x0_single)
            traj = x_seq[0, :, :2].cpu().numpy()

            # Plot the trajectory
            ax.plot(traj[:, 0], traj[:, 1], color=colors[j], linewidth=2.5, label=labels[j])

            # Plot the hill boundaries (only once per level so it doesn't clutter)
            if i == 0:
                for k in range(c_np.shape[0]):
                    ax.add_patch(plt.Circle((c_np[k, 0], c_np[k, 1]), r_val, color=colors[j], fill=False, alpha=0.5,
                                            linestyle="dotted"))

        ax.set_xlim(-start_box, start_box)
        ax.set_ylim(-start_box, start_box)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Start {i + 1}")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Contextual Adaptation to Changing Hill Sizes", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "context_adaptation.png"))
    plt.close(fig)
    print(f"Saved eval plots to {run_dir}")

# ---------------------------------------------------------------------------
# 5. MAIN TRAINING LOOP
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup Directories ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"landscape_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # --- Hyperparameters ---
    epochs = 150
    batch_size = 1024
    horizon = 80
    start_box = 3.0

    w_dim, u_dim, k = 4, 2, 3
    z_dim = 2 + 2 * k + k + k
    feat_dim = 16

    # --- Plants ---
    plant_nom = DoubleIntegratorNominal(dt=0.05, pre_kp=1.0, pre_kd=1.5)
    # The True plant has Unmodeled Wind to ensure w_t != 0 !
    plant_true = NoisyDoubleIntegratorTrue(dt=0.05, wind_x=0.3, wind_y=0.2)

    # --- PB Operator (No u_nominal!) ---
    mp = MpDeepSSM(w_dim, feat_dim, mode="loop", param="lru", n_layers=4, d_model=16, d_state=32, ff="GLU").to(device)
    mb = BoundedMLPOperator(w_dim=w_dim, z_dim=z_dim, r=u_dim, s=feat_dim, hidden_dim=64, bound_mode="softsign",
                            clamp_value=10.0).to(device)

    controller = PBController(
        plant=plant_nom, operator=FactorizedOperator(mp, mb),
        u_nominal=None, u_dim=u_dim, detach_state=False
    ).to(device)

    optimizer = optim.Adam(controller.parameters(), lr=2e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    # Fixed Centers for consistent learning
    fixed_centers = torch.tensor([[1.0, 0.0], [0.3, 0.8], [0.3, -0.8]], dtype=torch.float32)

    print(f"Starting Cost Landscape Training on {device} (u_nominal=None)...")

    # Generate a fixed validation set
    val_scen_cpu = sample_scenarios(batch_size=512, seed=999, k=k, r_min=0.3, r_max=0.6, fixed_centers=fixed_centers,
                                    start_box=(-start_box, start_box))
    val_scen_cpu, _ = ensure_challenging_starts(val_scen_cpu, start_box=(-start_box, start_box), min_fraction=0.7)
    val_scenario = type(val_scen_cpu)(start=val_scen_cpu.start.to(device), goal=val_scen_cpu.goal.to(device),
                                      centers=val_scen_cpu.centers.to(device), radii=val_scen_cpu.radii.to(device))

    for epoch in range(1, epochs + 1):
        controller.train()
        optimizer.zero_grad()

        # Curriculum: Hills grow wider over time
        progress = min(1.0, epoch / 100.0)
        r_min, r_max = 0.1 + 0.2 * progress, 0.2 + 0.5 * progress

        # Sample Training Set
        train_scen_cpu = sample_scenarios(batch_size=batch_size, seed=42 + epoch, k=k, r_min=r_min, r_max=r_max,
                                          fixed_centers=fixed_centers, start_box=(-start_box, start_box))
        train_scen_cpu, _ = ensure_challenging_starts(train_scen_cpu, start_box=(-start_box, start_box),
                                                      min_fraction=0.6)
        train_scenario = type(train_scen_cpu)(start=train_scen_cpu.start.to(device),
                                              goal=train_scen_cpu.goal.to(device),
                                              centers=train_scen_cpu.centers.to(device),
                                              radii=train_scen_cpu.radii.to(device))

        x0 = torch.cat([train_scenario.start, torch.zeros(batch_size, 2, device=device)], dim=-1).unsqueeze(1)

        def ctx_fn(x, t):
            return build_context(x, train_scenario)

        # Rollout (Notice: No artificial process noise injected, the wind handles w_t)
        x_seq, u_seq, w_seq = rollout_bptt(
            controller=controller, plant_true=plant_true, x0=x0,
            horizon=horizon, context_fn=ctx_fn, w0=x0
        )

        loss, parts = compute_loss(x_seq, u_seq, train_scenario)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            controller.eval()
            with torch.no_grad():
                val_x0 = torch.cat([val_scenario.start, torch.zeros(512, 2, device=device)], dim=-1).unsqueeze(1)

                def val_ctx(x, t): return build_context(x, val_scenario)

                val_x, val_u, _ = rollout_bptt(
                    controller=controller,
                    plant_true=plant_true,
                    x0=val_x0,
                    horizon=horizon,
                    context_fn=val_ctx,
                    w0=val_x0
                )
                val_loss, val_parts = compute_loss(val_x, val_u, val_scenario)

            print(
                f"Epoch {epoch:03d}/{epochs} | Train Loss: {loss.item():.2f} | Val Loss: {val_loss.item():.2f} | Val Term Dist: {val_parts['term_dist']:.3f} | Hill Cost: {val_parts['hill_cost']:.2f} | r_max: {r_max:.2f}")

    # Generate Final Plot
    print(f"Training Complete. Generating Landscape Plots in {run_dir}...")
    plot_landscape_and_trajectories(val_x, val_scenario, run_dir, start_box=start_box, show_plots=False)
    torch.save(controller.state_dict(), os.path.join(run_dir, "pb_model_landscape.pt"))

    generate_eval_plots(controller, plant_true, run_dir, start_box=start_box, device=device)


if __name__ == "__main__":
    main()