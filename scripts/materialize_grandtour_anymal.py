"""Materialize a small GrandTour ANYmal subset into NPZ episodes for Phase 2."""

from __future__ import annotations

import argparse
import logging
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import zarr

logger = logging.getLogger("olsd.materialize_grandtour_anymal")


def materialize_grandtour_anymal(
    source_dir: str | Path,
    output_dir: str | Path = "data/external/grandtour_anymal",
    max_episodes: int = 5,
    target_hz: float = 50.0,
    max_steps: int | None = 2048,
) -> list[Path]:
    """Convert a local GrandTour subset into NPZ episodes loadable by sim2real IO."""
    source = Path(source_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    mission_dirs = [path for path in sorted(source.iterdir()) if path.is_dir() and path.name != ".cache"]

    for mission_dir in mission_dirs[:max_episodes]:
        episode = _materialize_mission(
            mission_dir,
            target_hz=target_hz,
            max_steps=max_steps,
        )
        output_path = out_dir / f"grandtour_anymal_{mission_dir.name}.npz"
        np.savez_compressed(
            output_path,
            **episode,
        )
        saved.append(output_path)

    logger.info("Saved %d GrandTour ANYmal episodes to %s", len(saved), out_dir)
    return saved


def _materialize_mission(
    mission_dir: Path,
    target_hz: float = 50.0,
    max_steps: int | None = 2048,
) -> dict[str, np.ndarray | float | str]:
    """Extract the minimal proprioceptive ANYmal signals from one GrandTour mission."""
    data_dir = mission_dir / "data"
    actuator = _open_tar_zarr(data_dir / "anymal_state_actuator.tar", "anymal_state_actuator")
    state_est = _open_tar_zarr(data_dir / "anymal_state_state_estimator.tar", "anymal_state_state_estimator")
    imu = _open_tar_zarr(data_dir / "anymal_imu.tar", "anymal_imu")
    odom = _open_tar_zarr(data_dir / "anymal_state_odometry.tar", "anymal_state_odometry")

    try:
        timestamps = np.asarray(actuator["timestamp"], dtype=np.float64)
        joint_positions = _stack_joint_series(actuator, "state_joint_position", len(timestamps))
        joint_velocities = _stack_joint_series(actuator, "state_joint_velocity", len(timestamps))
        joint_torques = _stack_joint_series(actuator, "state_joint_torque", len(timestamps))
        actions = _stack_joint_series(actuator, "command_position", len(timestamps))

        state_timestamps = np.asarray(state_est["timestamp"], dtype=np.float64)
        base_position = _resample_rows(
            state_timestamps,
            np.asarray(state_est["pose_pos"], dtype=np.float64),
            timestamps,
        )
        base_orientation = _resample_rows(
            state_timestamps,
            np.asarray(state_est["pose_orien"], dtype=np.float64),
            timestamps,
        )
        base_velocity = _resample_rows(
            state_timestamps,
            np.asarray(state_est["twist_lin"], dtype=np.float64),
            timestamps,
        )
        base_angular_velocity = _resample_rows(
            state_timestamps,
            np.asarray(state_est["twist_ang"], dtype=np.float64),
            timestamps,
        )
        contact_binary = np.stack(
            [
                _resample_binary(state_timestamps, np.asarray(state_est[f"{foot}_contact"]), timestamps)
                for foot in ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
            ],
            axis=1,
        )

        imu_timestamps = np.asarray(imu["timestamp"], dtype=np.float64)
        imu_orientation = _resample_rows(
            imu_timestamps,
            np.asarray(imu["orien"], dtype=np.float64),
            timestamps,
        )
        imu_angular_velocity = _resample_rows(
            imu_timestamps,
            np.asarray(imu["ang_vel"], dtype=np.float64),
            timestamps,
        )
        imu_linear_acceleration = _resample_rows(
            imu_timestamps,
            np.asarray(imu["lin_acc"], dtype=np.float64),
            timestamps,
        )

        odom_timestamps = np.asarray(odom["timestamp"], dtype=np.float64)
        odom_position = _resample_rows(
            odom_timestamps,
            np.asarray(odom["pose_pos"], dtype=np.float64),
            timestamps,
        )
        odom_orientation = _resample_rows(
            odom_timestamps,
            np.asarray(odom["pose_orien"], dtype=np.float64),
            timestamps,
        )

        sampling_rate_hz = _infer_sampling_rate_hz(timestamps)
        episode = {
            "joint_positions": joint_positions.astype(np.float32),
            "joint_velocities": joint_velocities.astype(np.float32),
            "joint_torques": joint_torques.astype(np.float32),
            "actions": actions.astype(np.float32),
            "timestamps": timestamps.astype(np.float32),
            "sampling_rate_hz": float(sampling_rate_hz),
            "base_position": base_position.astype(np.float32),
            "base_velocity": base_velocity.astype(np.float32),
            "base_angular_velocity": base_angular_velocity.astype(np.float32),
            "imu_orientation": imu_orientation.astype(np.float32),
            "imu_angular_velocity": imu_angular_velocity.astype(np.float32),
            "imu_linear_acceleration": imu_linear_acceleration.astype(np.float32),
            "contact_binary": contact_binary.astype(np.float32),
            "state_estimator_position": base_position.astype(np.float32),
            "state_estimator_orientation": base_orientation.astype(np.float32),
            "odometry_position": odom_position.astype(np.float32),
            "odometry_orientation": odom_orientation.astype(np.float32),
            "external_dataset": "grandtour",
            "external_episode_id": mission_dir.name,
        }
        episode = _subsample_episode(episode, target_hz=target_hz)
        return _trim_episode(episode, max_steps=max_steps)
    finally:
        for group in (actuator, state_est, imu, odom):
            _cleanup_group(group)


def _open_tar_zarr(tar_path: Path, root_name: str):
    """Extract a tar-backed Zarr group to a temp directory and open it."""
    tempdir = tempfile.TemporaryDirectory()
    with tarfile.open(tar_path, "r") as archive:
        archive.extractall(tempdir.name)
    group = zarr.open_group(str(Path(tempdir.name) / root_name), mode="r")
    setattr(group, "_tempdir", tempdir)
    return group


def _cleanup_group(group) -> None:
    """Release a temp directory attached to a Zarr group."""
    tempdir = getattr(group, "_tempdir", None)
    if tempdir is not None:
        tempdir.cleanup()


def _stack_joint_series(group, suffix: str, n_steps: int) -> np.ndarray:
    """Stack the 12 actuator channels into a [T, 12] array."""
    columns = []
    for joint_idx in range(12):
        key = f"{joint_idx:02d}_{suffix}"
        values = np.asarray(group[key], dtype=np.float64)
        columns.append(values[:n_steps])
    return np.stack(columns, axis=1)


def _resample_rows(source_timestamps: np.ndarray, rows: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Resample a 2D time series onto the target timestamps."""
    clipped = np.clip(
        target_timestamps,
        float(source_timestamps[0]),
        float(source_timestamps[-1]),
    )
    if rows.ndim == 1:
        return np.interp(clipped, source_timestamps, rows).astype(np.float64)

    return np.stack(
        [np.interp(clipped, source_timestamps, rows[:, dim]) for dim in range(rows.shape[1])],
        axis=1,
    ).astype(np.float64)


def _resample_binary(source_timestamps: np.ndarray, values: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Nearest-neighbor resample for binary contact streams."""
    indices = np.searchsorted(source_timestamps, target_timestamps, side="left")
    indices = np.clip(indices, 0, len(source_timestamps) - 1)
    return values[indices].astype(np.float64)


def _infer_sampling_rate_hz(timestamps: np.ndarray) -> float:
    """Infer the nominal sampling rate from timestamps."""
    if len(timestamps) < 2:
        return 200.0
    dt = float(np.mean(np.diff(timestamps)))
    if dt <= 0:
        return 200.0
    return 1.0 / dt


def _subsample_episode(
    episode: dict[str, np.ndarray | float | str],
    target_hz: float,
) -> dict[str, np.ndarray | float | str]:
    """Subsample all time-aligned arrays to the requested rate."""
    timestamps = np.asarray(episode["timestamps"], dtype=np.float64)
    source_hz = float(episode["sampling_rate_hz"])
    if target_hz <= 0 or source_hz <= target_hz:
        episode["sampling_rate_hz"] = source_hz
        return episode

    step = max(1, int(round(source_hz / target_hz)))
    indices = np.arange(0, len(timestamps), step, dtype=np.int64)
    subsampled: dict[str, np.ndarray | float | str] = {}
    for key, value in episode.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == timestamps.shape:
            subsampled[key] = value[indices]
        else:
            subsampled[key] = value
    subsampled["sampling_rate_hz"] = float(target_hz)
    return subsampled


def _trim_episode(
    episode: dict[str, np.ndarray | float | str],
    max_steps: int | None,
) -> dict[str, np.ndarray | float | str]:
    """Keep only the leading calibration window when requested."""
    if max_steps is None:
        return episode

    timestamps = np.asarray(episode["timestamps"])
    if len(timestamps) <= max_steps:
        return episode

    trimmed: dict[str, np.ndarray | float | str] = {}
    for key, value in episode.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == timestamps.shape:
            trimmed[key] = value[:max_steps]
        else:
            trimmed[key] = value
    return trimmed


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize GrandTour ANYmal subset as NPZ episodes")
    parser.add_argument("--source-dir", default="data/external/grandtour_subset")
    parser.add_argument("--output-dir", default="data/external/grandtour_anymal")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=2048)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    materialize_grandtour_anymal(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        target_hz=args.target_hz,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
