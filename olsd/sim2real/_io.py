"""Internal loaders for sim-to-real workflows."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from olsd.pipeline.ingest import load_robot_by_id
from olsd.schema import (
    Action,
    ControlMode,
    DataSource,
    Episode,
    EpisodeMetadata,
    Observation,
    Step,
    TerrainSpec,
    TerrainType,
)
from olsd.sdk.loader import load


GO1_OBS_SLICES = {
    "height_quat": slice(0, 5),
    "joint_positions": slice(5, 17),
    "base_velocity": slice(17, 20),
    "base_angular_velocity": slice(20, 23),
    "joint_velocities": slice(23, 35),
}


def load_episodes_from_path(
    path: str | Path,
    robot_id: str | None = None,
    terrain_type: TerrainType = TerrainType.FLAT,
) -> list[Episode]:
    """Load OLSD Episodes from a dataset directory or a directory of NPZ files."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Episode source not found: {source}")

    if source.is_dir() and (source / "meta").exists():
        return load(source).episodes

    if source.is_file() and source.suffix == ".npz":
        return [_episode_from_npz(source, robot_id=robot_id, terrain_type=terrain_type)]

    if source.is_dir():
        npz_files = sorted(source.glob("*.npz"))
        if npz_files:
            return [
                _episode_from_npz(npz_path, robot_id=robot_id, terrain_type=terrain_type)
                for npz_path in npz_files
            ]

    raise ValueError(f"Unsupported episode source: {source}")


def load_mjcf_xml(path_or_xml: str | Path | None) -> str | None:
    """Load MJCF XML from a file path or return an XML string as-is."""
    if path_or_xml is None:
        return None

    raw = str(path_or_xml)
    if raw.lstrip().startswith("<mujoco"):
        return raw

    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"MJCF path not found: {path}")
    return path.read_text(encoding="utf-8")


def save_json(data: dict, output_path: str | Path) -> Path:
    """Persist a JSON-serializable dict."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return output


def clone_episode_with_arrays(
    reference_episode: Episode,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    actions: np.ndarray | None = None,
) -> Episode:
    """Create a new Episode by replacing the core motion arrays."""
    metadata = deepcopy(reference_episode.metadata)
    steps: list[Step] = []

    for idx, ref_step in enumerate(reference_episode.steps):
        obs = deepcopy(ref_step.observation)
        obs.joint_positions = joint_positions[idx].tolist()
        obs.joint_velocities = joint_velocities[idx].tolist()

        if actions is None:
            action_values = ref_step.action.values
        else:
            action_values = actions[idx].tolist()

        steps.append(
            Step(
                observation=obs,
                action=Action(
                    values=action_values,
                    control_mode=ref_step.action.control_mode,
                ),
                reward=ref_step.reward,
                done=ref_step.done,
                truncated=ref_step.truncated,
                timestamp=ref_step.timestamp,
                info=deepcopy(ref_step.info),
            )
        )

    return Episode(
        episode_id=reference_episode.episode_id,
        steps=steps,
        metadata=metadata,
    )


def _episode_from_npz(
    npz_path: Path,
    robot_id: str | None = None,
    terrain_type: TerrainType = TerrainType.FLAT,
) -> Episode:
    """Convert a supported NPZ file to an Episode."""
    data = np.load(npz_path, allow_pickle=True)

    if "joint_positions" in data:
        joint_positions = np.asarray(data["joint_positions"], dtype=np.float32)
        joint_velocities = np.asarray(
            data.get("joint_velocities", np.zeros_like(joint_positions)),
            dtype=np.float32,
        )
        actions = np.asarray(data.get("actions", joint_positions), dtype=np.float32)
        rewards = np.asarray(data.get("rewards", np.zeros(len(joint_positions))), dtype=np.float32)
        timestamps = _build_timestamps(
            data.get("timestamps"),
            len(joint_positions),
            hz=float(data.get("sampling_rate_hz", 50.0)),
        )
        robot_spec = _resolve_robot_spec(robot_id, joint_positions.shape[1])
        metadata = EpisodeMetadata(
            robot=robot_spec,
            terrain=TerrainSpec(terrain_type=terrain_type),
            source=(
                DataSource.HARDWARE
                if "external_dataset" in data
                else DataSource.SIMULATION
            ),
            sampling_rate_hz=_infer_sampling_rate(timestamps),
            external_dataset=_optional_str(data, "external_dataset"),
            external_episode_id=_optional_str(data, "external_episode_id"),
        )
        extra_arrays = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in [
                "joint_torques",
                "base_position",
                "base_velocity",
                "base_angular_velocity",
                "imu_orientation",
                "imu_angular_velocity",
                "imu_linear_acceleration",
                "contact_binary",
            ]
            if key in data
        }
        return _build_episode(
            episode_id=npz_path.stem,
            metadata=metadata,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            actions=actions,
            rewards=rewards,
            timestamps=timestamps,
            extra_arrays=extra_arrays,
        )

    if "observations" in data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] < GO1_OBS_SLICES["joint_velocities"].stop:
            raise ValueError(f"Unsupported observations array in {npz_path}")

        joint_positions = observations[:, GO1_OBS_SLICES["joint_positions"]]
        joint_velocities = observations[:, GO1_OBS_SLICES["joint_velocities"]]
        base_velocity = observations[:, GO1_OBS_SLICES["base_velocity"]]
        base_angular_velocity = observations[:, GO1_OBS_SLICES["base_angular_velocity"]]
        height_quat = observations[:, GO1_OBS_SLICES["height_quat"]]
        actions = np.asarray(data.get("actions", joint_positions), dtype=np.float32)
        rewards = np.asarray(data.get("rewards", np.zeros(len(joint_positions))), dtype=np.float32)
        timestamps = _build_timestamps(data.get("timestamps"), len(joint_positions), hz=50.0)
        robot_spec = _resolve_robot_spec(robot_id or "go1", joint_positions.shape[1])
        metadata = EpisodeMetadata(
            robot=robot_spec,
            terrain=TerrainSpec(terrain_type=terrain_type),
            source=DataSource.SIMULATION,
            simulator="mujoco",
            sampling_rate_hz=_infer_sampling_rate(timestamps),
        )

        steps: list[Step] = []
        for idx in range(len(joint_positions)):
            quat = height_quat[idx, 1:5].tolist()
            steps.append(
                Step(
                    observation=Observation(
                        joint_positions=joint_positions[idx].tolist(),
                        joint_velocities=joint_velocities[idx].tolist(),
                        imu_orientation=quat,
                        base_position=[0.0, 0.0, float(height_quat[idx, 0])],
                        base_velocity=base_velocity[idx].tolist(),
                        base_angular_velocity=base_angular_velocity[idx].tolist(),
                    ),
                    action=Action(
                        values=actions[idx].tolist(),
                        control_mode=ControlMode.POSITION,
                    ),
                    reward=float(rewards[idx]),
                    done=idx == (len(joint_positions) - 1),
                    timestamp=float(timestamps[idx]),
                )
            )

        return Episode(
            episode_id=npz_path.stem,
            steps=steps,
            metadata=metadata,
        )

    raise ValueError(f"Unsupported NPZ layout in {npz_path}")


def _build_episode(
    episode_id: str,
    metadata: EpisodeMetadata,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    timestamps: np.ndarray,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> Episode:
    """Build an Episode from raw arrays."""
    extras = extra_arrays or {}
    steps: list[Step] = []
    for idx in range(len(joint_positions)):
        observation = Observation(
            joint_positions=joint_positions[idx].tolist(),
            joint_velocities=joint_velocities[idx].tolist(),
        )
        if "joint_torques" in extras:
            observation.joint_torques = extras["joint_torques"][idx].tolist()
        if "base_position" in extras:
            observation.base_position = extras["base_position"][idx].tolist()
        if "base_velocity" in extras:
            observation.base_velocity = extras["base_velocity"][idx].tolist()
        if "base_angular_velocity" in extras:
            observation.base_angular_velocity = extras["base_angular_velocity"][idx].tolist()
        if "imu_orientation" in extras:
            observation.imu_orientation = extras["imu_orientation"][idx].tolist()
        if "imu_angular_velocity" in extras:
            observation.imu_angular_velocity = extras["imu_angular_velocity"][idx].tolist()
        if "imu_linear_acceleration" in extras:
            observation.imu_linear_acceleration = extras["imu_linear_acceleration"][idx].tolist()
        if "contact_binary" in extras:
            observation.contact_binary = extras["contact_binary"][idx].tolist()
        steps.append(
            Step(
                observation=observation,
                action=Action(
                    values=actions[idx].tolist(),
                    control_mode=ControlMode.POSITION,
                ),
                reward=float(rewards[idx]),
                done=idx == (len(joint_positions) - 1),
                timestamp=float(timestamps[idx]),
            )
        )

    return Episode(episode_id=episode_id, steps=steps, metadata=metadata)


def _build_timestamps(
    timestamps: np.ndarray | None,
    n_steps: int,
    hz: float,
) -> np.ndarray:
    """Infer timestamps when they are not stored."""
    if timestamps is not None:
        return np.asarray(timestamps, dtype=np.float32)
    dt = 1.0 / hz
    return np.arange(n_steps, dtype=np.float32) * dt


def _infer_sampling_rate(timestamps: np.ndarray) -> float:
    """Infer the sampling rate from timestamps."""
    if len(timestamps) < 2:
        return 50.0
    dt = float(np.mean(np.diff(timestamps)))
    if dt <= 0:
        return 50.0
    return 1.0 / dt


def _resolve_robot_spec(robot_id: str | None, n_joints: int):
    """Resolve a robot config, falling back to a minimal spec."""
    if robot_id:
        try:
            return load_robot_by_id(robot_id)
        except FileNotFoundError:
            pass

    from olsd.schema import Morphology, RobotSpec

    return RobotSpec(
        robot_id=robot_id or "unknown_robot",
        robot_name=robot_id or "Unknown Robot",
        morphology=Morphology.QUADRUPED if n_joints >= 8 else Morphology.OTHER,
        n_joints=n_joints,
        n_actuators=n_joints,
        mass_kg=0.0,
    )


def _optional_str(data, key: str) -> str | None:
    """Decode optional scalar string values stored in NPZ files."""
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
