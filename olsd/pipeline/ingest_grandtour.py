"""
GrandTour ANYmal-D Ingestor — Convert Zarr/HuggingFace data to OLSD Episodes.

Source: https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset
License: CC-BY-4.0 ✅

The GrandTour dataset is the largest open quadruped dataset, collected on an
ANYmal-D with 12 proprioceptive joints across 49 environments (indoor, outdoor,
alpine, forest, industrial).

We ingest ONLY kinematics + terrain labels — cameras/LiDAR are skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from olsd.schema import (
    Action,
    ControlMode,
    DataSource,
    Episode,
    EpisodeMetadata,
    GaitType,
    Morphology,
    Observation,
    RobotSpec,
    Step,
    TerrainSpec,
    TerrainType,
)

logger = logging.getLogger(__name__)


# ANYmal-D joint names (HAA=Hip Abduction/Adduction, HFE=Hip Flexion/Extension, KFE=Knee)
ANYMAL_JOINTS = [
    "LF_HAA", "LF_HFE", "LF_KFE",  # Left Front
    "RF_HAA", "RF_HFE", "RF_KFE",  # Right Front
    "LH_HAA", "LH_HFE", "LH_KFE",  # Left Hind
    "RH_HAA", "RH_HFE", "RH_KFE",  # Right Hind
]

ANYMAL_D_SPEC = RobotSpec(
    robot_id="anymal_d",
    robot_name="ANYmal D",
    morphology=Morphology.QUADRUPED,
    n_joints=12,
    n_actuators=12,
    mass_kg=50.0,
    n_legs=4,
    dof_per_leg=3,
    manufacturer="ANYbotics",
    description="Research quadruped with 3-DOF legs. Data from GrandTour (ETH Zurich).",
)

# Default terrain label mapping based on sequence naming conventions
DEFAULT_TERRAIN_LABELS: dict[str, TerrainType] = {
    "indoor": TerrainType.FLAT,
    "office": TerrainType.FLAT,
    "parking": TerrainType.CONCRETE,
    "forest": TerrainType.ROUGH,
    "alpine": TerrainType.ROCKY,
    "grass": TerrainType.GRASS,
    "gravel": TerrainType.GRAVEL,
    "stairs": TerrainType.STAIRS,
    "slope": TerrainType.SLOPE,
    "snow": TerrainType.SNOW,
}


def _infer_terrain(sequence_name: str, terrain_labels: dict[str, TerrainType] | None = None) -> TerrainType:
    """Infer terrain type from sequence name using keyword matching."""
    labels = {**DEFAULT_TERRAIN_LABELS, **(terrain_labels or {})}
    name_lower = sequence_name.lower()
    for keyword, terrain in labels.items():
        if keyword in name_lower:
            return terrain
    return TerrainType.CUSTOM


def from_grandtour(
    data_path: str | Path,
    max_episodes: int | None = None,
    terrain_labels: dict[str, TerrainType] | None = None,
    subsample_hz: float = 50.0,
    source_hz: float = 200.0,
    robot: RobotSpec | None = None,
) -> list[Episode]:
    """
    Convert GrandTour data to OLSD Episodes.

    Supports two input formats:
      1. Zarr store (from HuggingFace datasets download)
      2. Directory of .npz files (pre-extracted kinematics)

    Field mapping (kinematics only — cameras/LiDAR skipped):
        joint/position  (12 values)  -> observation.joint_positions
        joint/velocity  (12 values)  -> observation.joint_velocities
        joint/effort    (12 values)  -> observation.joint_torques
        imu/orientation (4 values)   -> observation.imu_orientation [w,x,y,z]
        imu/angular_vel (3 values)   -> observation.imu_angular_velocity
        imu/linear_acc  (3 values)   -> observation.imu_linear_acceleration
        state_est/pose  (3+ values)  -> observation.base_position [x,y,z]

    Args:
        data_path: Path to Zarr store, HF cache directory, or .npz directory.
        max_episodes: Maximum number of episodes to load (None = all).
        terrain_labels: Optional override mapping sequence names to terrain types.
        subsample_hz: Target output frequency (default 50Hz).
        source_hz: Source data frequency (default 200Hz).
        robot: Override robot spec (default: ANYmal-D).

    Returns:
        List of OLSD Episode objects.
    """
    path = Path(data_path)
    _robot = robot or ANYMAL_D_SPEC
    episodes: list[Episode] = []
    subsample_step = max(1, int(source_hz / subsample_hz))

    if path.suffix == ".zarr" or (path / ".zarray").exists() or (path / ".zgroup").exists():
        episodes = _from_zarr(path, _robot, terrain_labels, subsample_step, max_episodes)
    elif any(path.glob("*.npz")):
        episodes = _from_npz_dir(path, _robot, terrain_labels, subsample_step, max_episodes)
    else:
        # Try loading as HuggingFace dataset cache
        episodes = _from_hf_cache(path, _robot, terrain_labels, subsample_step, max_episodes)

    logger.info(f"GrandTour: loaded {len(episodes)} episodes, "
                f"{sum(e.n_steps for e in episodes)} total steps")
    return episodes


def _from_zarr(
    zarr_path: Path,
    robot: RobotSpec,
    terrain_labels: dict[str, TerrainType] | None,
    subsample_step: int,
    max_episodes: int | None,
) -> list[Episode]:
    """Load from a Zarr store."""
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr is required for GrandTour ingestion. Install: pip install zarr"
        )

    store = zarr.open(str(zarr_path), mode="r")
    episodes = []

    # GrandTour Zarr structure varies; handle common layouts
    sequence_keys = [k for k in store.keys() if not k.startswith(".")]

    for seq_name in sequence_keys:
        if max_episodes and len(episodes) >= max_episodes:
            break

        seq = store[seq_name]
        ep = _zarr_sequence_to_episode(seq, seq_name, robot, terrain_labels, subsample_step)
        if ep is not None:
            episodes.append(ep)

    return episodes


def _zarr_sequence_to_episode(
    seq: Any,
    seq_name: str,
    robot: RobotSpec,
    terrain_labels: dict[str, TerrainType] | None,
    subsample_step: int,
) -> Episode | None:
    """Convert a single Zarr sequence group to an Episode."""
    try:
        # Try common GrandTour Zarr key patterns
        jp = _get_zarr_array(seq, ["joint/position", "joint_position", "joint_positions",
                                    "proprioception/joint_position"])
        jv = _get_zarr_array(seq, ["joint/velocity", "joint_velocity", "joint_velocities",
                                    "proprioception/joint_velocity"])

        if jp is None or jv is None:
            logger.warning(f"Skipping sequence '{seq_name}': missing joint data")
            return None

        # Optional fields
        jt = _get_zarr_array(seq, ["joint/effort", "joint_torque", "joint_torques",
                                    "proprioception/joint_effort"])
        imu_quat = _get_zarr_array(seq, ["imu/orientation", "imu_orientation",
                                          "proprioception/imu_orientation"])
        imu_gyro = _get_zarr_array(seq, ["imu/angular_velocity", "imu_angular_velocity",
                                          "proprioception/imu_angular_velocity"])
        imu_acc = _get_zarr_array(seq, ["imu/linear_acceleration", "imu_linear_acceleration",
                                         "proprioception/imu_linear_acceleration"])
        base_pos = _get_zarr_array(seq, ["state_estimator/pose", "base_position",
                                          "proprioception/base_position"])

    except Exception as e:
        logger.warning(f"Error reading sequence '{seq_name}': {e}")
        return None

    n_steps = len(jp)
    steps = []
    dt = subsample_step / 200.0  # assuming 200Hz source

    for i in range(0, n_steps, subsample_step):
        obs = Observation(
            joint_positions=jp[i].tolist(),
            joint_velocities=jv[i].tolist() if jv is not None else [0.0] * len(jp[i]),
        )

        # Optional fields
        if jt is not None and i < len(jt):
            obs.joint_torques = jt[i].tolist()
        if imu_quat is not None and i < len(imu_quat):
            obs.imu_orientation = imu_quat[i].tolist()
        if imu_gyro is not None and i < len(imu_gyro):
            obs.imu_angular_velocity = imu_gyro[i].tolist()
        if imu_acc is not None and i < len(imu_acc):
            obs.imu_linear_acceleration = imu_acc[i].tolist()
        if base_pos is not None and i < len(base_pos):
            pos = base_pos[i]
            obs.base_position = pos[:3].tolist() if len(pos) >= 3 else pos.tolist()

        # Use joint torques as "actions" for hardware data (control commands sent to actuators)
        act_values = jt[i].tolist() if (jt is not None and i < len(jt)) else jp[i].tolist()
        action = Action(values=act_values, control_mode=ControlMode.TORQUE)

        step = Step(
            observation=obs,
            action=action,
            reward=None,  # no reward for hardware data
            done=False,
            timestamp=i * (1.0 / 200.0),
        )
        steps.append(step)

    if not steps:
        return None

    terrain = _infer_terrain(seq_name, terrain_labels)
    metadata = EpisodeMetadata(
        robot=robot,
        terrain=TerrainSpec(terrain_type=terrain),
        source=DataSource.HARDWARE,
        sampling_rate_hz=200.0 / subsample_step,
        external_dataset="grandtour",
        external_episode_id=seq_name,
        contributor="ETH Zurich / Robotic Systems Lab",
        institution="ETH Zurich",
        license="CC-BY-4.0",
    )

    return Episode(
        episode_id=str(uuid4()),
        steps=steps,
        metadata=metadata,
    )


def _from_npz_dir(
    npz_dir: Path,
    robot: RobotSpec,
    terrain_labels: dict[str, TerrainType] | None,
    subsample_step: int,
    max_episodes: int | None,
) -> list[Episode]:
    """Load from a directory of pre-extracted .npz files."""
    episodes = []
    npz_files = sorted(npz_dir.glob("*.npz"))

    for npz_path in npz_files:
        if max_episodes and len(episodes) >= max_episodes:
            break

        data = np.load(npz_path, allow_pickle=True)
        seq_name = npz_path.stem

        jp = data.get("joint_positions", data.get("joint_position"))
        jv = data.get("joint_velocities", data.get("joint_velocity"))

        if jp is None:
            logger.warning(f"Skipping {npz_path}: no joint_positions found")
            continue

        n_steps = len(jp)
        steps = []

        for i in range(0, n_steps, subsample_step):
            obs = Observation(
                joint_positions=jp[i].tolist(),
                joint_velocities=jv[i].tolist() if jv is not None else [0.0] * len(jp[i]),
            )

            if "joint_torques" in data and i < len(data["joint_torques"]):
                obs.joint_torques = data["joint_torques"][i].tolist()
            if "imu_orientation" in data and i < len(data["imu_orientation"]):
                obs.imu_orientation = data["imu_orientation"][i].tolist()
            if "base_position" in data and i < len(data["base_position"]):
                obs.base_position = data["base_position"][i][:3].tolist()

            act_values = jp[i].tolist()
            action = Action(values=act_values, control_mode=ControlMode.POSITION)

            step = Step(
                observation=obs,
                action=action,
                timestamp=i * (1.0 / 200.0),
            )
            steps.append(step)

        if not steps:
            continue

        terrain = _infer_terrain(seq_name, terrain_labels)
        metadata = EpisodeMetadata(
            robot=robot,
            terrain=TerrainSpec(terrain_type=terrain),
            source=DataSource.HARDWARE,
            sampling_rate_hz=200.0 / subsample_step,
            external_dataset="grandtour",
            external_episode_id=seq_name,
        )

        episodes.append(Episode(
            episode_id=str(uuid4()),
            steps=steps,
            metadata=metadata,
        ))

    return episodes


def _from_hf_cache(
    cache_dir: Path,
    robot: RobotSpec,
    terrain_labels: dict[str, TerrainType] | None,
    subsample_step: int,
    max_episodes: int | None,
) -> list[Episode]:
    """Load from a HuggingFace datasets cache directory (Arrow format)."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(cache_dir))
    except Exception:
        logger.warning(f"Could not load HF cache from {cache_dir}")
        return []

    episodes = []
    for idx, row in enumerate(ds):
        if max_episodes and len(episodes) >= max_episodes:
            break

        ep = _hf_row_to_episode(row, idx, robot, terrain_labels, subsample_step)
        if ep is not None:
            episodes.append(ep)

    return episodes


def _hf_row_to_episode(
    row: dict,
    idx: int,
    robot: RobotSpec,
    terrain_labels: dict[str, TerrainType] | None,
    subsample_step: int,
) -> Episode | None:
    """Convert a HuggingFace dataset row to an Episode."""
    # HF format depends on how GrandTour structures their dataset
    # Try common column patterns
    for jp_key in ["joint_position", "joint_positions", "proprioception.joint_position"]:
        if jp_key in row:
            jp = np.array(row[jp_key])
            break
    else:
        return None

    jv_key = jp_key.replace("position", "velocity")
    jv = np.array(row.get(jv_key, np.zeros_like(jp)))

    # If data is 1D (single timestep), wrap in 2D
    if jp.ndim == 1:
        jp = jp.reshape(1, -1)
        jv = jv.reshape(1, -1)

    n_steps = len(jp)
    steps = []
    seq_name = row.get("sequence_name", row.get("name", f"seq_{idx}"))

    for i in range(0, n_steps, subsample_step):
        obs = Observation(
            joint_positions=jp[i].tolist(),
            joint_velocities=jv[i].tolist(),
        )
        action = Action(values=jp[i].tolist(), control_mode=ControlMode.POSITION)
        step = Step(observation=obs, action=action, timestamp=i * (1.0 / 200.0))
        steps.append(step)

    if not steps:
        return None

    terrain = _infer_terrain(str(seq_name), terrain_labels)
    metadata = EpisodeMetadata(
        robot=robot,
        terrain=TerrainSpec(terrain_type=terrain),
        source=DataSource.HARDWARE,
        sampling_rate_hz=200.0 / subsample_step,
        external_dataset="grandtour",
        external_episode_id=str(seq_name),
    )

    return Episode(episode_id=str(uuid4()), steps=steps, metadata=metadata)


def _get_zarr_array(group: Any, possible_keys: list[str]) -> np.ndarray | None:
    """Try multiple key patterns to find an array in a Zarr group."""
    for key in possible_keys:
        parts = key.split("/")
        current = group
        try:
            for part in parts:
                current = current[part]
            return np.array(current)
        except (KeyError, TypeError, IndexError):
            continue
    return None
