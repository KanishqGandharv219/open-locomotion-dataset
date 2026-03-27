"""
OLSD Export Pipeline — Convert Episodes to various output formats.

Supported targets:
  - Parquet / Arrow  (LeRobot v3-compatible, HF-native)
  - Hugging Face Dataset object
  - HDF5 (for interop with D4RL tooling)
  - JSON (for debugging / small datasets)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from olsd.schema import DatasetInfo, Episode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet / LeRobot v3 export
# ---------------------------------------------------------------------------


def to_parquet(
    episodes: list[Episode],
    output_dir: str | Path,
    dataset_info: DatasetInfo | None = None,
    chunk_size: int = 10000,
) -> Path:
    """
    Export episodes to Parquet files in a LeRobot v3-compatible layout.

    Output structure:
        output_dir/
        ├── data/
        │   ├── chunk-000.parquet
        │   ├── chunk-001.parquet
        │   └── ...
        ├── meta/
        │   ├── info.json
        │   ├── episodes.json
        │   └── stats.json
        └── README.md
    """
    out = Path(output_dir)
    data_dir = out / "data"
    meta_dir = out / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Flatten episodes into rows
    rows = []
    episode_records = []
    global_idx = 0

    for ep_idx, episode in enumerate(episodes):
        ep_start = global_idx
        for frame_idx, step in enumerate(episode.steps):
            row = {
                "index": global_idx,
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "timestamp": step.timestamp,
                # Observation
                "observation.joint_positions": step.observation.joint_positions,
                "observation.joint_velocities": step.observation.joint_velocities,
                # Action
                "action": step.action.values,
                "action.control_mode": step.action.control_mode.value,
                # Outcome
                "reward": step.reward or 0.0,
                "done": step.done,
                "truncated": step.truncated,
                # Metadata (per-step for queryability)
                "robot_id": episode.metadata.robot.robot_id,
                "morphology": episode.metadata.robot.morphology.value,
                "terrain_type": episode.metadata.terrain.terrain_type.value,
                "source": episode.metadata.source.value,
            }

            # Optional fields
            if step.observation.joint_torques:
                row["observation.joint_torques"] = step.observation.joint_torques
            if step.observation.imu_orientation:
                row["observation.imu_orientation"] = step.observation.imu_orientation
            if step.observation.contact_binary is not None:
                row["observation.contact_binary"] = step.observation.contact_binary
            if step.observation.base_position:
                row["observation.base_position"] = step.observation.base_position
            if step.observation.base_velocity:
                row["observation.base_velocity"] = step.observation.base_velocity

            rows.append(row)
            global_idx += 1

        episode_records.append({
            "episode_index": ep_idx,
            "episode_id": episode.episode_id,
            "length": len(episode.steps),
            "index_from": ep_start,
            "index_to": global_idx - 1,
            "robot_id": episode.metadata.robot.robot_id,
            "morphology": episode.metadata.robot.morphology.value,
            "terrain_type": episode.metadata.terrain.terrain_type.value,
            "gait_type": episode.metadata.gait_type.value if episode.metadata.gait_type else None,
            "success": episode.metadata.success,
            "source": episode.metadata.source.value,
        })

    # Write Parquet chunks
    if rows:
        # Group rows into chunks
        for chunk_idx in range(0, len(rows), chunk_size):
            chunk_rows = rows[chunk_idx : chunk_idx + chunk_size]
            table = pa.Table.from_pylist(chunk_rows)
            chunk_path = data_dir / f"chunk-{chunk_idx // chunk_size:03d}.parquet"
            pq.write_table(table, chunk_path)

    # Write metadata
    info = dataset_info or DatasetInfo()
    info.total_episodes = len(episodes)
    info.total_steps = global_idx
    info.robots = list(set(r["robot_id"] for r in episode_records))
    info.terrains = list(set(r["terrain_type"] for r in episode_records))
    info.morphologies = list(set(r["morphology"] for r in episode_records))
    info.updated_at = datetime.now()

    # info.json
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info.model_dump(mode="json"), f, indent=2, default=str)

    # episodes.json
    with open(meta_dir / "episodes.json", "w") as f:
        json.dump(episode_records, f, indent=2)

    # stats.json — compute per-feature min/max/mean/std
    stats = _compute_stats(episodes)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Exported {len(episodes)} episodes ({global_idx} steps) to {out}")
    return out


# ---------------------------------------------------------------------------
# Hugging Face Dataset export
# ---------------------------------------------------------------------------


def to_hf_dataset(episodes: list[Episode]) -> Any:
    """
    Convert episodes to a Hugging Face Dataset object.

    Returns a datasets.Dataset that can be pushed to Hub.
    """
    from datasets import Dataset

    records = []
    for ep_idx, episode in enumerate(episodes):
        for frame_idx, step in enumerate(episode.steps):
            records.append({
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "timestamp": step.timestamp,
                "observation.joint_positions": step.observation.joint_positions,
                "observation.joint_velocities": step.observation.joint_velocities,
                "action": step.action.values,
                "reward": step.reward or 0.0,
                "done": step.done,
                "robot_id": episode.metadata.robot.robot_id,
                "morphology": episode.metadata.robot.morphology.value,
                "terrain_type": episode.metadata.terrain.terrain_type.value,
                "source": episode.metadata.source.value,
            })

    dataset = Dataset.from_list(records)
    logger.info(f"Created HF Dataset with {len(records)} rows")
    return dataset


# ---------------------------------------------------------------------------
# HDF5 export (D4RL-compatible)
# ---------------------------------------------------------------------------


def to_hdf5(
    episodes: list[Episode],
    output_path: str | Path,
) -> Path:
    """Export episodes to HDF5 in D4RL-style flat format."""
    import h5py

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_acts = []
    all_rews = []
    all_terms = []
    all_timeouts = []

    for episode in episodes:
        data = episode.to_numpy()
        all_obs.append(data["joint_positions"])
        all_acts.append(data["actions"])
        all_rews.append(data["rewards"])

        terms = data["dones"].copy()
        timeouts = np.zeros_like(terms)
        if not terms[-1]:
            timeouts[-1] = True  # mark timeout if not terminated
        all_terms.append(terms)
        all_timeouts.append(timeouts)

    with h5py.File(path, "w") as f:
        f["observations"] = np.concatenate(all_obs)
        f["actions"] = np.concatenate(all_acts)
        f["rewards"] = np.concatenate(all_rews)
        f["terminals"] = np.concatenate(all_terms)
        f["timeouts"] = np.concatenate(all_timeouts)

    logger.info(f"Exported {len(episodes)} episodes to HDF5: {path}")
    return path


# ---------------------------------------------------------------------------
# JSON export (debug / small datasets)
# ---------------------------------------------------------------------------


def to_json(
    episodes: list[Episode],
    output_path: str | Path,
    indent: int = 2,
) -> Path:
    """Export episodes to a JSON file (for small datasets / debugging)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [ep.model_dump(mode="json") for ep in episodes]
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)

    logger.info(f"Exported {len(episodes)} episodes to JSON: {path}")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_stats(episodes: list[Episode]) -> dict:
    """Compute per-feature statistics across all episodes, grouped by robot."""
    from collections import defaultdict

    # Group by robot_id since different robots have different dimensions
    by_robot: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"jp": [], "jv": [], "act": [], "rew": []}
    )

    for ep in episodes:
        rid = ep.metadata.robot.robot_id
        data = ep.to_numpy()
        by_robot[rid]["jp"].append(data["joint_positions"])
        by_robot[rid]["jv"].append(data["joint_velocities"])
        by_robot[rid]["act"].append(data["actions"])
        by_robot[rid]["rew"].append(data["rewards"])

    def _stats(arrays: list[np.ndarray]) -> dict:
        if not arrays:
            return {}
        concat = np.concatenate(arrays)
        return {
            "min": concat.min(axis=0).tolist() if concat.ndim > 1 else float(concat.min()),
            "max": concat.max(axis=0).tolist() if concat.ndim > 1 else float(concat.max()),
            "mean": concat.mean(axis=0).tolist() if concat.ndim > 1 else float(concat.mean()),
            "std": concat.std(axis=0).tolist() if concat.ndim > 1 else float(concat.std()),
        }

    stats: dict = {}
    for rid, arrays in by_robot.items():
        stats[rid] = {
            "observation.joint_positions": _stats(arrays["jp"]),
            "observation.joint_velocities": _stats(arrays["jv"]),
            "action": _stats(arrays["act"]),
            "reward": _stats(arrays["rew"]),
        }

    # Also compute global scalar stats (reward is always 1-D, safe to concat)
    all_rew = []
    for arrays in by_robot.values():
        all_rew.extend(arrays["rew"])
    if all_rew:
        stats["_global"] = {"reward": _stats(all_rew)}

    return stats
