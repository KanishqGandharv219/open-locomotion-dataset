"""
OLSD Visualization — Plot trajectories, gait diagrams, and metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from olsd.schema import Episode
from olsd.pipeline.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

OLSD_COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "bg": "#0f172a",
    "surface": "#1e293b",
    "text": "#f1f5f9",
}


def _apply_style():
    """Apply OLSD dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": OLSD_COLORS["bg"],
        "axes.facecolor": OLSD_COLORS["surface"],
        "axes.edgecolor": "#475569",
        "axes.labelcolor": OLSD_COLORS["text"],
        "text.color": OLSD_COLORS["text"],
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 10,
    })


# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------


def plot_trajectory(
    episode: Episode,
    joints: list[int] | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot joint positions and velocities over time.

    Args:
        episode: Episode to plot
        joints: Which joint indices to include (None = all)
        show: Whether to call plt.show()
        save_path: Path to save figure
    """
    _apply_style()
    data = episode.to_numpy()
    t = data["timestamps"]
    jp = data["joint_positions"]
    jv = data["joint_velocities"]

    n_joints = jp.shape[1]
    joints = joints or list(range(min(n_joints, 6)))  # Max 6 joints for readability

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    cmap = plt.cm.viridis(np.linspace(0.2, 0.8, len(joints)))

    # Joint positions
    for i, j_idx in enumerate(joints):
        axes[0].plot(t, jp[:, j_idx], color=cmap[i], label=f"Joint {j_idx}", alpha=0.8, linewidth=1.5)
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title(
        f"Trajectory: {episode.metadata.robot.robot_name} | "
        f"{episode.metadata.terrain.terrain_type.value} | "
        f"{episode.n_steps} steps",
        fontsize=12, fontweight="bold",
    )
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.7)
    axes[0].grid(True)

    # Joint velocities
    for i, j_idx in enumerate(joints):
        axes[1].plot(t, jv[:, j_idx], color=cmap[i], label=f"Joint {j_idx}", alpha=0.8, linewidth=1.5)
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_actions(
    episode: Episode,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot action commands over time."""
    _apply_style()
    data = episode.to_numpy()
    t = data["timestamps"]
    actions = data["actions"]

    n_act = min(actions.shape[1], 6)
    cmap = plt.cm.plasma(np.linspace(0.2, 0.8, n_act))

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(n_act):
        ax.plot(t, actions[:, i], color=cmap[i], label=f"Act {i}", alpha=0.8, linewidth=1.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action value")
    ax.set_title("Actions over time", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_phase_portrait(
    episode: Episode,
    joint_idx: int = 0,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot phase portrait (position vs velocity) for a single joint."""
    _apply_style()
    data = episode.to_numpy()
    jp = data["joint_positions"][:, joint_idx]
    jv = data["joint_velocities"][:, joint_idx]
    t = data["timestamps"]

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        jp, jv, c=t, cmap="viridis", s=3, alpha=0.7,
    )
    ax.plot(jp, jv, color=OLSD_COLORS["primary"], alpha=0.3, linewidth=0.5)

    ax.set_xlabel(f"Joint {joint_idx} Position (rad)")
    ax.set_ylabel(f"Joint {joint_idx} Velocity (rad/s)")
    ax.set_title(f"Phase Portrait — Joint {joint_idx}", fontsize=12, fontweight="bold")
    plt.colorbar(scatter, label="Time (s)")
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_rewards(
    episode: Episode,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot reward signal and cumulative reward over time."""
    _apply_style()
    data = episode.to_numpy()
    t = data["timestamps"]
    rewards = data["rewards"]
    cum_reward = np.cumsum(rewards)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, rewards, color=OLSD_COLORS["accent"], linewidth=1.2)
    axes[0].fill_between(t, rewards, alpha=0.2, color=OLSD_COLORS["accent"])
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward Signal", fontweight="bold")
    axes[0].grid(True)

    axes[1].plot(t, cum_reward, color=OLSD_COLORS["success"], linewidth=2)
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"Total Return: {cum_reward[-1]:.1f}", fontweight="bold")
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Gait diagram
# ---------------------------------------------------------------------------


def plot_gait_diagram(
    episode: Episode,
    n_feet: int = 4,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot footfall timing diagram (gait diagram).

    Uses contact_binary if available, otherwise estimates from joint velocities.
    """
    _apply_style()
    data = episode.to_numpy()
    t = data["timestamps"]

    # Try contact data first
    contacts = []
    for step in episode.steps:
        if step.observation.contact_binary is not None:
            contacts.append(step.observation.contact_binary)

    if contacts:
        contact_array = np.array(contacts)  # (T, n_feet)
    else:
        # Estimate contacts from joint velocity (low velocity ≈ stance)
        jv = data["joint_velocities"]
        n_joints = jv.shape[1]
        joints_per_leg = max(1, n_joints // n_feet)
        contact_array = np.zeros((len(t), n_feet))
        for leg in range(min(n_feet, n_joints // max(1, joints_per_leg))):
            start = leg * joints_per_leg
            end = start + joints_per_leg
            leg_vel = np.sum(np.abs(jv[:, start:end]), axis=1)
            threshold = np.percentile(leg_vel, 40)
            contact_array[:, leg] = (leg_vel < threshold).astype(float)

    foot_names = ["FR", "FL", "RR", "RL"][:n_feet]

    fig, ax = plt.subplots(figsize=(12, 3))

    for foot_idx in range(min(n_feet, contact_array.shape[1])):
        stance = contact_array[:, foot_idx]
        # Draw filled bars for stance phases
        for i in range(len(t) - 1):
            if stance[i] > 0.5:
                ax.barh(
                    foot_idx, t[i + 1] - t[i], left=t[i], height=0.6,
                    color=OLSD_COLORS["primary"], alpha=0.8,
                )

    ax.set_yticks(range(len(foot_names)))
    ax.set_yticklabels(foot_names)
    ax.set_xlabel("Time (s)")
    ax.set_title("Gait Diagram (shaded = stance)", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Dataset-level plots
# ---------------------------------------------------------------------------


def plot_dataset_overview(
    episodes: list[Episode],
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot dataset overview: distribution of robots, terrains, and episode lengths."""
    _apply_style()
    from collections import Counter

    robots = Counter(ep.metadata.robot.robot_id for ep in episodes)
    terrains = Counter(ep.metadata.terrain.terrain_type.value for ep in episodes)
    lengths = [ep.n_steps for ep in episodes]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Robot distribution
    bars = axes[0].barh(list(robots.keys()), list(robots.values()), color=OLSD_COLORS["primary"])
    axes[0].set_title("Episodes per Robot", fontweight="bold")
    axes[0].set_xlabel("Count")

    # Terrain distribution
    bars = axes[1].barh(list(terrains.keys()), list(terrains.values()), color=OLSD_COLORS["accent"])
    axes[1].set_title("Episodes per Terrain", fontweight="bold")
    axes[1].set_xlabel("Count")

    # Episode length distribution
    axes[2].hist(lengths, bins=30, color=OLSD_COLORS["secondary"], alpha=0.8, edgecolor="#475569")
    axes[2].set_title("Episode Length Distribution", fontweight="bold")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Count")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_metrics_comparison(
    episodes: list[Episode],
    group_by: str = "robot_id",
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot metrics comparison across groups (robots or terrains)."""
    _apply_style()
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for ep in episodes:
        if group_by == "robot_id":
            key = ep.metadata.robot.robot_id
        elif group_by == "terrain":
            key = ep.metadata.terrain.terrain_type.value
        else:
            key = "all"
        m = compute_metrics(ep)
        groups[key].append(m)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    keys = sorted(groups.keys())
    x = np.arange(len(keys))

    # Energy per meter
    means = [np.mean([m.energy_per_meter for m in groups[k]]) for k in keys]
    stds = [np.std([m.energy_per_meter for m in groups[k]]) for k in keys]
    axes[0].bar(x, means, yerr=stds, color=OLSD_COLORS["warning"], alpha=0.8, capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(keys, rotation=45, ha="right")
    axes[0].set_title("Energy per Meter (J/m)", fontweight="bold")

    # Smoothness
    means = [np.mean([m.smoothness_index for m in groups[k]]) for k in keys]
    stds = [np.std([m.smoothness_index for m in groups[k]]) for k in keys]
    axes[1].bar(x, means, yerr=stds, color=OLSD_COLORS["accent"], alpha=0.8, capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(keys, rotation=45, ha="right")
    axes[1].set_title("Smoothness Index (lower=better)", fontweight="bold")

    # Stride frequency
    means = [np.mean([m.stride_frequency for m in groups[k]]) for k in keys]
    stds = [np.std([m.stride_frequency for m in groups[k]]) for k in keys]
    axes[2].bar(x, means, yerr=stds, color=OLSD_COLORS["primary"], alpha=0.8, capsize=3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(keys, rotation=45, ha="right")
    axes[2].set_title("Stride Frequency (Hz)", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig
