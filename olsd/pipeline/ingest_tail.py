"""
TAIL Dataset Ingestor — Deformable/Sandy Terrain Locomotion Data.

Thin ingestor: kinematics + terrain labels ONLY.
Skips: RGB, LiDAR, depth, point clouds (too large, not needed for locomotion benchmark).

TAIL provides unique terrain diversity (sand, grass, gravel, deformable surfaces)
that no other open dataset covers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import numpy as np

from olsd.schema import (
    Action,
    ControlMode,
    DataSource,
    Episode,
    EpisodeMetadata,
    Morphology,
    Observation,
    RobotSpec,
    Step,
    TerrainSpec,
    TerrainType,
)

logger = logging.getLogger(__name__)

# TAIL terrain type mapping from sequence names
TAIL_TERRAIN_MAP: dict[str, TerrainType] = {
    "sand": TerrainType.SAND,
    "sandy": TerrainType.SAND,
    "grass": TerrainType.GRASS,
    "gravel": TerrainType.GRAVEL,
    "mud": TerrainType.MUD,
    "soft": TerrainType.CUSTOM,
    "deformable": TerrainType.CUSTOM,
    "hard": TerrainType.CONCRETE,
    "indoor": TerrainType.FLAT,
    "asphalt": TerrainType.CONCRETE,
}


def _infer_tail_terrain(name: str) -> TerrainType:
    """Infer terrain type from TAIL sequence name."""
    name_lower = name.lower()
    for keyword, terrain in TAIL_TERRAIN_MAP.items():
        if keyword in name_lower:
            return terrain
    return TerrainType.CUSTOM


def from_tail(
    data_path: str | Path,
    robot_id: str = "tail_quadruped",
    max_episodes: int | None = None,
    sampling_rate_hz: float = 50.0,
) -> list[Episode]:
    """
    Convert TAIL dataset to OLSD Episodes — kinematics + terrain labels ONLY.

    Skips all perception modalities (RGB, LiDAR, depth) to keep data lightweight.

    Supports:
      - Directory of .npz files (pre-extracted kinematics)
      - Directory of .csv files (tabular trajectory logs)

    Args:
        data_path: Path to TAIL data directory.
        robot_id: Robot identifier for this TAIL subset.
        max_episodes: Maximum number of episodes to load.
        sampling_rate_hz: Target sampling rate.

    Returns:
        List of OLSD Episode objects with terrain labels.
    """
    path = Path(data_path)
    episodes: list[Episode] = []

    robot = RobotSpec(
        robot_id=robot_id,
        robot_name="TAIL Quadruped",
        morphology=Morphology.QUADRUPED,
        n_joints=12,  # typical quadruped
        n_actuators=12,
        mass_kg=25.0,
        n_legs=4,
        dof_per_leg=3,
        description="Quadruped from TAIL deformable-terrain dataset.",
    )

    if any(path.glob("*.npz")):
        episodes = _from_npz(path, robot, max_episodes, sampling_rate_hz)
    elif any(path.glob("*.csv")):
        episodes = _from_csv(path, robot, max_episodes, sampling_rate_hz)
    elif any(path.rglob("*.npz")):
        # Subdirectories may contain npz files per terrain type
        for subdir in sorted(path.iterdir()):
            if max_episodes and len(episodes) >= max_episodes:
                break
            if subdir.is_dir() and any(subdir.glob("*.npz")):
                remaining = (max_episodes - len(episodes)) if max_episodes else None
                sub_eps = _from_npz(subdir, robot, remaining, sampling_rate_hz)
                episodes.extend(sub_eps)
    else:
        logger.warning(f"No recognized TAIL data found in {path}")

    logger.info(f"TAIL: loaded {len(episodes)} episodes, "
                f"{sum(e.n_steps for e in episodes)} total steps")
    return episodes


def _from_npz(
    npz_dir: Path,
    robot: RobotSpec,
    max_episodes: int | None,
    sampling_rate_hz: float,
) -> list[Episode]:
    """Load TAIL episodes from .npz files."""
    episodes = []
    terrain_name = npz_dir.stem  # use directory name as terrain hint

    for npz_path in sorted(npz_dir.glob("*.npz")):
        if max_episodes and len(episodes) >= max_episodes:
            break

        data = np.load(npz_path, allow_pickle=True)

        # Try common key patterns for joint data
        jp = None
        for key in ["joint_positions", "joint_position", "q", "qpos", "positions"]:
            if key in data:
                jp = data[key]
                break

        if jp is None:
            # Fall back: use first array with correct shape
            for key in data.files:
                arr = data[key]
                if arr.ndim == 2 and arr.shape[1] in (12, 8, 6):
                    jp = arr
                    break

        if jp is None:
            logger.debug(f"Skipping {npz_path}: no joint data found")
            continue

        # Joint velocities
        jv = None
        for key in ["joint_velocities", "joint_velocity", "qd", "qvel", "velocities"]:
            if key in data:
                jv = data[key]
                break

        if jv is None:
            # Compute via finite differences
            jv = np.zeros_like(jp)
            jv[1:] = (jp[1:] - jp[:-1]) * sampling_rate_hz

        n_joints = jp.shape[1]
        robot_adjusted = RobotSpec(
            robot_id=robot.robot_id,
            robot_name=robot.robot_name,
            morphology=robot.morphology,
            n_joints=n_joints,
            n_actuators=n_joints,
            mass_kg=robot.mass_kg,
            n_legs=robot.n_legs,
            dof_per_leg=n_joints // (robot.n_legs or 4),
        )

        steps = []
        dt = 1.0 / sampling_rate_hz
        for i in range(len(jp)):
            obs = Observation(
                joint_positions=jp[i].tolist(),
                joint_velocities=jv[i].tolist(),
            )
            action = Action(values=jp[i].tolist(), control_mode=ControlMode.POSITION)
            step = Step(observation=obs, action=action, timestamp=i * dt)
            steps.append(step)

        if not steps:
            continue

        seq_name = f"{terrain_name}/{npz_path.stem}"
        terrain = _infer_tail_terrain(seq_name)

        metadata = EpisodeMetadata(
            robot=robot_adjusted,
            terrain=TerrainSpec(
                terrain_type=terrain,
                description=f"TAIL terrain: {terrain_name}",
            ),
            source=DataSource.HARDWARE,
            sampling_rate_hz=sampling_rate_hz,
            external_dataset="tail",
            external_episode_id=seq_name,
        )

        episodes.append(Episode(
            episode_id=str(uuid4()),
            steps=steps,
            metadata=metadata,
        ))

    return episodes


def _from_csv(
    csv_dir: Path,
    robot: RobotSpec,
    max_episodes: int | None,
    sampling_rate_hz: float,
) -> list[Episode]:
    """Load TAIL episodes from CSV files."""
    import pandas as pd

    episodes = []
    terrain_name = csv_dir.stem

    for csv_path in sorted(csv_dir.glob("*.csv")):
        if max_episodes and len(episodes) >= max_episodes:
            break

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"Failed to read {csv_path}: {e}")
            continue

        # Identify joint columns by pattern matching
        jp_cols = [c for c in df.columns if any(k in c.lower() for k in ["pos", "q_", "joint"])]
        jv_cols = [c for c in df.columns if any(k in c.lower() for k in ["vel", "qd", "dq"])]

        if not jp_cols:
            logger.debug(f"Skipping {csv_path}: no joint columns found")
            continue

        jp = df[jp_cols].values
        jv = df[jv_cols].values if jv_cols else np.gradient(jp, 1.0 / sampling_rate_hz, axis=0)

        n_joints = jp.shape[1]
        steps = []
        dt = 1.0 / sampling_rate_hz

        for i in range(len(jp)):
            obs = Observation(
                joint_positions=jp[i].tolist(),
                joint_velocities=jv[i].tolist(),
            )
            action = Action(values=jp[i].tolist(), control_mode=ControlMode.POSITION)
            step = Step(observation=obs, action=action, timestamp=i * dt)
            steps.append(step)

        if not steps:
            continue

        terrain = _infer_tail_terrain(terrain_name)
        metadata = EpisodeMetadata(
            robot=RobotSpec(
                robot_id=robot.robot_id, robot_name=robot.robot_name,
                morphology=robot.morphology, n_joints=n_joints,
                n_actuators=n_joints, mass_kg=robot.mass_kg,
            ),
            terrain=TerrainSpec(terrain_type=terrain),
            source=DataSource.HARDWARE,
            sampling_rate_hz=sampling_rate_hz,
            external_dataset="tail",
            external_episode_id=f"{terrain_name}/{csv_path.stem}",
        )

        episodes.append(Episode(
            episode_id=str(uuid4()), steps=steps, metadata=metadata,
        ))

    return episodes
