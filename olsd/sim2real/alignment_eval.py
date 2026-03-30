"""Sim-vs-real trajectory alignment metrics."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from olsd.schema import Episode
from olsd.sim2real._io import load_episodes_from_path, save_json

logger = logging.getLogger(__name__)


@dataclass
class AlignmentReport:
    """Aggregate alignment metrics across paired episodes."""

    joint_rmse: float
    per_joint_rmse: list[float]
    velocity_correlation: float
    trajectory_dtw: float
    n_pairs: int
    mean_shared_steps: float
    mean_shared_joints: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_episode_alignment(real_episode: Episode, sim_episode: Episode) -> dict[str, float | list[float]]:
    """Evaluate alignment for a single pair of episodes."""
    real_pos, sim_pos, real_vel, sim_vel = _shared_arrays(real_episode, sim_episode)

    pos_error = real_pos - sim_pos
    per_joint_rmse = np.sqrt(np.mean(np.square(pos_error), axis=0))
    joint_rmse = float(np.mean(per_joint_rmse))

    real_vel_flat = real_vel.reshape(-1)
    sim_vel_flat = sim_vel.reshape(-1)
    velocity_correlation = _pearson(real_vel_flat, sim_vel_flat)
    trajectory_dtw = _dtw_distance(real_pos, sim_pos)

    return {
        "joint_rmse": joint_rmse,
        "per_joint_rmse": per_joint_rmse.tolist(),
        "velocity_correlation": velocity_correlation,
        "trajectory_dtw": trajectory_dtw,
        "shared_steps": int(real_pos.shape[0]),
        "shared_joints": int(real_pos.shape[1]),
    }


def evaluate_alignment(
    real_episodes: list[Episode],
    sim_episodes: list[Episode],
) -> dict[str, float | list[float]]:
    """Compare paired real and simulated episodes on shared time/joint support."""
    if not real_episodes or not sim_episodes:
        raise ValueError("Both real_episodes and sim_episodes must be non-empty")

    n_pairs = min(len(real_episodes), len(sim_episodes))
    episode_reports = [
        evaluate_episode_alignment(real_episodes[idx], sim_episodes[idx])
        for idx in range(n_pairs)
    ]

    max_joints = max(int(report["shared_joints"]) for report in episode_reports)
    per_joint_buckets: list[list[float]] = [[] for _ in range(max_joints)]
    for report in episode_reports:
        for joint_idx, value in enumerate(report["per_joint_rmse"]):
            per_joint_buckets[joint_idx].append(float(value))

    per_joint_rmse = [
        float(np.mean(bucket)) if bucket else 0.0
        for bucket in per_joint_buckets
    ]

    aggregate = AlignmentReport(
        joint_rmse=float(np.mean([report["joint_rmse"] for report in episode_reports])),
        per_joint_rmse=per_joint_rmse,
        velocity_correlation=float(
            np.mean([report["velocity_correlation"] for report in episode_reports])
        ),
        trajectory_dtw=float(np.mean([report["trajectory_dtw"] for report in episode_reports])),
        n_pairs=n_pairs,
        mean_shared_steps=float(np.mean([report["shared_steps"] for report in episode_reports])),
        mean_shared_joints=float(np.mean([report["shared_joints"] for report in episode_reports])),
    )
    return aggregate.to_dict()


def save_alignment_report(report: dict, output_path: str | Path) -> Path:
    """Persist an alignment report as JSON."""
    return save_json(report, output_path)


def _shared_arrays(
    real_episode: Episode,
    sim_episode: Episode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice both episodes to their shared time horizon and DOF."""
    real_data = real_episode.to_numpy()
    sim_data = sim_episode.to_numpy()

    shared_steps = min(len(real_data["joint_positions"]), len(sim_data["joint_positions"]))
    shared_joints = min(
        real_data["joint_positions"].shape[1],
        sim_data["joint_positions"].shape[1],
    )

    real_pos = real_data["joint_positions"][:shared_steps, :shared_joints]
    sim_pos = sim_data["joint_positions"][:shared_steps, :shared_joints]
    real_vel = real_data["joint_velocities"][:shared_steps, :shared_joints]
    sim_vel = sim_data["joint_velocities"][:shared_steps, :shared_joints]
    return real_pos, sim_pos, real_vel, sim_vel


def _pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Compute a robust Pearson correlation."""
    if lhs.size == 0 or rhs.size == 0:
        return 0.0
    lhs_std = float(np.std(lhs))
    rhs_std = float(np.std(rhs))
    if lhs_std < 1e-8 and rhs_std < 1e-8:
        return 1.0
    if lhs_std < 1e-8 or rhs_std < 1e-8:
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _dtw_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Dynamic time warping distance using per-step Euclidean distances."""
    n_steps, m_steps = lhs.shape[0], rhs.shape[0]
    cost = np.full((n_steps + 1, m_steps + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n_steps + 1):
        for j in range(1, m_steps + 1):
            point_cost = float(np.linalg.norm(lhs[i - 1] - rhs[j - 1]))
            cost[i, j] = point_cost + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

    return float(cost[n_steps, m_steps] / max(n_steps + m_steps, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sim-vs-real alignment")
    parser.add_argument("--real-data", required=True, help="Path to real episodes")
    parser.add_argument("--sim-data", required=True, help="Path to simulated episodes")
    parser.add_argument(
        "--output",
        default="results/sim2real_report.json",
        help="Output JSON path",
    )
    parser.add_argument("--robot", default=None, help="Optional robot id hint for NPZ sources")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    real_episodes = load_episodes_from_path(args.real_data, robot_id=args.robot)
    sim_episodes = load_episodes_from_path(args.sim_data, robot_id=args.robot)
    report = evaluate_alignment(real_episodes, sim_episodes)
    output = save_alignment_report(report, args.output)

    logger.info(
        "Alignment report saved to %s (joint_rmse=%.4f, velocity_corr=%.4f, dtw=%.4f)",
        output,
        report["joint_rmse"],
        report["velocity_correlation"],
        report["trajectory_dtw"],
    )


if __name__ == "__main__":
    main()
