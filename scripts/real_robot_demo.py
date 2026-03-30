"""Real robot simulation demo powered by the reusable Go1 sim2real env."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from olsd.pipeline.metrics import compute_metrics
from olsd.schema import GaitType
from olsd.sim2real.go1_env import (
    BoundingGaitController,
    TrottingGaitController,
    rollout_controller,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("olsd.real_robot_demo")


def main() -> None:
    logger.info("=" * 60)
    logger.info("  OLSD Real Robot Simulation - Unitree Go1 Quadruped")
    logger.info("=" * 60)

    gaits = {
        "trot": TrottingGaitController(frequency=2.0),
        "trot_fast": TrottingGaitController(frequency=3.0, amplitude_thigh=0.5),
        "bound": BoundingGaitController(frequency=2.5),
    }

    episodes = []
    for gait_name, controller in gaits.items():
        logger.info("Rolling out %s gait", gait_name)
        for idx in range(5):
            episode = rollout_controller(controller, n_steps=500)
            episode.episode_id = f"go1_{gait_name}_{idx}"
            episode.metadata.gait_type = GaitType.TROT if "trot" in gait_name else GaitType.BOUND
            episodes.append(episode)
            metrics = compute_metrics(episode)
            logger.info(
                "  Episode %d: steps=%d reward=%.1f stride=%.2fHz success=%s",
                idx,
                episode.n_steps,
                float(sum(step.reward or 0.0 for step in episode.steps)),
                metrics.stride_frequency,
                episode.metadata.success,
            )

    _save_visualization(episodes, gaits)
    _export_npz(episodes)

    total_steps = sum(ep.n_steps for ep in episodes)
    logger.info("Exported %d episodes (%d steps total)", len(episodes), total_steps)


def _save_visualization(episodes: list, gaits: dict) -> None:
    colors = {"trot": "#8b7cf6", "trot_fast": "#06b6d4", "bound": "#10b981"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("OLSD - Unitree Go1 Quadruped Simulation", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for gait_name in gaits:
        gait_episodes = [ep for ep in episodes if gait_name in ep.episode_id]
        rewards = [[step.reward or 0.0 for step in ep.steps] for ep in gait_episodes]
        for values in rewards:
            ax.plot(values, color=colors[gait_name], alpha=0.3, linewidth=0.8)
        min_len = min(len(values) for values in rewards)
        mean_rewards = np.mean([values[:min_len] for values in rewards], axis=0)
        ax.plot(mean_rewards, color=colors[gait_name], linewidth=2, label=gait_name)
    ax.set_title("Reward per Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    obs = np.array([step.observation.joint_positions for step in episodes[0].steps])
    for idx, label in enumerate(["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf"]):
        style = "--" if label.startswith("FL_") else "-"
        ax.plot(obs[:, idx], label=label, linewidth=1.2, linestyle=style)
    ax.set_title("Joint Positions - FR vs FL")
    ax.set_xlabel("Step")
    ax.set_ylabel("Joint Angle (rad)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for gait_name in gaits:
        gait_episodes = [ep for ep in episodes if gait_name in ep.episode_id]
        distances = [ep.steps[-1].observation.base_position[0] if ep.steps[-1].observation.base_position else 0.0 for ep in gait_episodes]
        x_positions = [x + list(gaits.keys()).index(gait_name) * 0.25 for x in range(len(gait_episodes))]
        ax.bar(x_positions, distances, width=0.25, color=colors[gait_name], label=gait_name, alpha=0.85)
    ax.set_title("Forward Distance per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    metric_labels = ["Reward", "Stride", "Length", "Success"]
    for gait_name in gaits:
        gait_episodes = [ep for ep in episodes if gait_name in ep.episode_id]
        rewards = [sum(step.reward or 0.0 for step in ep.steps) for ep in gait_episodes]
        strides = [compute_metrics(ep).stride_frequency for ep in gait_episodes]
        lengths = [ep.n_steps / 500.0 for ep in gait_episodes]
        success = [1.0 if ep.metadata.success else 0.0 for ep in gait_episodes]
        reward_scale = max(1.0, max(abs(value) for value in rewards))
        stride_scale = max(0.1, max(strides))
        values = [
            max(0.0, np.mean(rewards)) / reward_scale,
            np.mean(strides) / stride_scale,
            np.mean(lengths),
            np.mean(success),
        ]
        ax.bar(
            [x + list(gaits.keys()).index(gait_name) * 0.25 for x in range(4)],
            values,
            width=0.25,
            color=colors[gait_name],
            label=gait_name,
            alpha=0.85,
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_title("Normalized Gait Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path("results/go1_simulation.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved visualization to %s", out_path)


def _export_npz(episodes: list) -> None:
    export_dir = Path("data/go1_episodes")
    export_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        observations = np.array(
            [
                step.observation.base_position[2:3]
                + step.observation.imu_orientation
                + step.observation.joint_positions
                + step.observation.base_velocity
                + step.observation.base_angular_velocity
                + step.observation.joint_velocities
                for step in episode.steps
            ],
            dtype=np.float32,
        )
        actions = np.array([step.action.values for step in episode.steps], dtype=np.float32)
        rewards = np.array([step.reward or 0.0 for step in episode.steps], dtype=np.float32)
        timestamps = np.array([step.timestamp for step in episode.steps], dtype=np.float32)
        np.savez_compressed(
            export_dir / f"{episode.episode_id}.npz",
            observations=observations,
            actions=actions,
            rewards=rewards,
            timestamps=timestamps,
        )


if __name__ == "__main__":
    main()
