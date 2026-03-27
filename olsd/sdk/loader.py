"""
OLSD SDK — Dataset loader for local and Hugging Face-hosted datasets.

Usage:
    import olsd
    dataset = olsd.load("./data/olsd-v0.1")
    dataset = olsd.load("open-locomotion-skills/olsd-v0.1", streaming=True)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq

from olsd.schema import (
    DatasetInfo,
    Episode,
    EpisodeMetadata,
    Morphology,
    TerrainType,
)

logger = logging.getLogger(__name__)


class OLSDDataset:
    """
    In-memory or streaming view over an OLSD dataset.

    Provides filtering, iteration, and conversion utilities.
    """

    def __init__(
        self,
        episodes: list[Episode] | None = None,
        info: DatasetInfo | None = None,
        path: str | Path | None = None,
    ):
        self._episodes = episodes or []
        self.info = info or DatasetInfo()
        self.path = Path(path) if path else None

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> Episode:
        return self._episodes[idx]

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)

    @property
    def episodes(self) -> list[Episode]:
        return self._episodes

    def filter(
        self,
        robot_id: str | None = None,
        morphology: str | Morphology | None = None,
        terrain: str | TerrainType | None = None,
        source: str | None = None,
        success: bool | None = None,
        min_steps: int | None = None,
        max_steps: int | None = None,
    ) -> "OLSDDataset":
        """Filter episodes by metadata criteria. Returns a new OLSDDataset."""
        filtered = self._episodes

        if robot_id is not None:
            filtered = [ep for ep in filtered if ep.metadata.robot.robot_id == robot_id]

        if morphology is not None:
            morph_val = morphology if isinstance(morphology, str) else morphology.value
            filtered = [ep for ep in filtered if ep.metadata.robot.morphology.value == morph_val]

        if terrain is not None:
            terrain_val = terrain if isinstance(terrain, str) else terrain.value
            filtered = [ep for ep in filtered if ep.metadata.terrain.terrain_type.value == terrain_val]

        if source is not None:
            filtered = [ep for ep in filtered if ep.metadata.source.value == source]

        if success is not None:
            filtered = [ep for ep in filtered if ep.metadata.success == success]

        if min_steps is not None:
            filtered = [ep for ep in filtered if ep.n_steps >= min_steps]

        if max_steps is not None:
            filtered = [ep for ep in filtered if ep.n_steps <= max_steps]

        return OLSDDataset(episodes=filtered, info=self.info, path=self.path)

    def summary(self) -> dict[str, Any]:
        """Get a summary of the dataset."""
        robots = set()
        morphologies = set()
        terrains = set()
        sources = set()
        total_steps = 0
        success_count = 0

        for ep in self._episodes:
            robots.add(ep.metadata.robot.robot_id)
            morphologies.add(ep.metadata.robot.morphology.value)
            terrains.add(ep.metadata.terrain.terrain_type.value)
            sources.add(ep.metadata.source.value)
            total_steps += ep.n_steps
            if ep.metadata.success:
                success_count += 1

        return {
            "total_episodes": len(self._episodes),
            "total_steps": total_steps,
            "robots": sorted(robots),
            "morphologies": sorted(morphologies),
            "terrains": sorted(terrains),
            "sources": sorted(sources),
            "success_rate": success_count / len(self._episodes) if self._episodes else 0.0,
        }

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert entire dataset to numpy arrays (concatenated across episodes)."""
        all_jp, all_jv, all_act, all_rew = [], [], [], []

        for ep in self._episodes:
            data = ep.to_numpy()
            all_jp.append(data["joint_positions"])
            all_jv.append(data["joint_velocities"])
            all_act.append(data["actions"])
            all_rew.append(data["rewards"])

        return {
            "joint_positions": np.concatenate(all_jp) if all_jp else np.array([]),
            "joint_velocities": np.concatenate(all_jv) if all_jv else np.array([]),
            "actions": np.concatenate(all_act) if all_act else np.array([]),
            "rewards": np.concatenate(all_rew) if all_rew else np.array([]),
        }

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
    ) -> tuple["OLSDDataset", "OLSDDataset", "OLSDDataset"]:
        """Split dataset into train/val/test by episodes."""
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self._episodes))

        n_train = int(len(indices) * train)
        n_val = int(len(indices) * val)

        train_eps = [self._episodes[i] for i in indices[:n_train]]
        val_eps = [self._episodes[i] for i in indices[n_train:n_train + n_val]]
        test_eps = [self._episodes[i] for i in indices[n_train + n_val:]]

        return (
            OLSDDataset(episodes=train_eps, info=self.info, path=self.path),
            OLSDDataset(episodes=val_eps, info=self.info, path=self.path),
            OLSDDataset(episodes=test_eps, info=self.info, path=self.path),
        )


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------


def load(source: str | Path, streaming: bool = False) -> OLSDDataset:
    """
    Load an OLSD dataset from a local path or Hugging Face repo.

    Args:
        source: Local directory path or HF repo ID (e.g. "org/dataset-name")
        streaming: If True and source is HF, use streaming mode
    """
    source_str = str(source)

    # Check if it's a HF repo ID (contains / but isn't a local path)
    if "/" in source_str and not Path(source_str).exists():
        return _load_from_hf(source_str, streaming=streaming)
    else:
        return _load_from_local(Path(source_str))


def _load_from_local(path: Path) -> OLSDDataset:
    """Load dataset from local Parquet files."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    # Load metadata
    info = DatasetInfo()
    meta_dir = path / "meta"
    if (meta_dir / "info.json").exists():
        with open(meta_dir / "info.json") as f:
            info_dict = json.load(f)
            info = DatasetInfo(**info_dict)

    # Load episodes from JSON if available (small datasets)
    json_path = path / "episodes.json"
    if json_path.exists():
        with open(json_path) as f:
            ep_dicts = json.load(f)
        episodes = [Episode(**d) for d in ep_dicts]
        return OLSDDataset(episodes=episodes, info=info, path=path)

    # Load from Parquet (larger datasets)
    data_dir = path / "data"
    if data_dir.exists():
        parquet_files = sorted(data_dir.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading {len(parquet_files)} parquet files from {data_dir}")
            episodes = _parquet_to_episodes(parquet_files, meta_dir)
            return OLSDDataset(episodes=episodes, info=info, path=path)

    logger.warning(f"No data files found in {path}")
    return OLSDDataset(info=info, path=path)


def _load_from_hf(repo_id: str, streaming: bool = False) -> OLSDDataset:
    """Load dataset from Hugging Face Hub."""
    from datasets import load_dataset

    logger.info(f"Loading dataset from Hugging Face: {repo_id}")
    hf_dataset = load_dataset(repo_id, streaming=streaming)

    if streaming:
        logger.info("Streaming mode: episodes will be loaded on-demand")
        # For streaming, return empty dataset with info
        return OLSDDataset(info=DatasetInfo(name=repo_id))

    # Convert HF dataset to episodes
    # This is a simplified conversion — full implementation would
    # reconstruct Episode objects from the flat tabular format
    logger.info(f"Loaded dataset with {len(hf_dataset)} rows")
    return OLSDDataset(info=DatasetInfo(name=repo_id))


def _parquet_to_episodes(
    parquet_files: list[Path],
    meta_dir: Path | None = None,
) -> list[Episode]:
    """Reconstruct Episode objects from Parquet files + episode metadata."""
    # Load episode metadata if available
    episode_meta = {}
    if meta_dir and (meta_dir / "episodes.json").exists():
        with open(meta_dir / "episodes.json") as f:
            for rec in json.load(f):
                episode_meta[rec["episode_index"]] = rec

    # Read all parquet files into a combined table
    import pandas as pd
    dfs = [pd.read_parquet(f) for f in parquet_files]
    if not dfs:
        return []
    df = pd.concat(dfs, ignore_index=True)

    # Group by episode_index and reconstruct
    from olsd.schema import (
        Action, ControlMode, DataSource, Observation, RobotSpec, Step, TerrainSpec,
    )

    episodes = []
    for ep_idx, group in df.groupby("episode_index"):
        group = group.sort_values("frame_index")

        steps = []
        for _, row in group.iterrows():
            step = Step(
                observation=Observation(
                    joint_positions=row["observation.joint_positions"],
                    joint_velocities=row["observation.joint_velocities"],
                ),
                action=Action(
                    values=row["action"],
                    control_mode=ControlMode(row.get("action.control_mode", "torque")),
                ),
                reward=float(row.get("reward", 0.0)),
                done=bool(row.get("done", False)),
                truncated=bool(row.get("truncated", False)),
                timestamp=float(row.get("timestamp", 0.0)),
            )
            steps.append(step)

        # Build metadata
        meta = episode_meta.get(ep_idx, {})
        robot = RobotSpec(
            robot_id=str(row.get("robot_id", "unknown")),
            robot_name=str(row.get("robot_id", "unknown")),
            morphology=Morphology(row.get("morphology", "other")),
            n_joints=len(steps[0].observation.joint_positions) if steps else 0,
            n_actuators=len(steps[0].action.values) if steps else 0,
            mass_kg=0.0,
        )

        metadata = EpisodeMetadata(
            robot=robot,
            terrain=TerrainSpec(terrain_type=TerrainType(row.get("terrain_type", "flat"))),
            source=DataSource(row.get("source", "simulation")),
            success=meta.get("success", True),
        )

        episodes.append(Episode(
            episode_id=meta.get("episode_id", f"ep_{ep_idx}"),
            steps=steps,
            metadata=metadata,
        ))

    return episodes
