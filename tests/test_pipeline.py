"""Tests for OLSD data pipeline (validate + metrics)."""

import numpy as np
import pytest

from olsd.pipeline.metrics import compute_metrics
from olsd.pipeline.validate import validate_episode
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


@pytest.fixture
def sample_episode():
    """Create a sample episode with sinusoidal joint patterns."""
    robot = RobotSpec(
        robot_id="test_bot", robot_name="Test Bot",
        morphology=Morphology.QUADRUPED, n_joints=4, n_actuators=4, mass_kg=5.0,
    )
    steps = []
    for i in range(100):
        t = i * 0.02  # 50 Hz
        jp = [np.sin(2 * np.pi * 2.0 * t + p) * 0.5 for p in [0, np.pi/2, np.pi, 3*np.pi/2]]
        jv = [np.cos(2 * np.pi * 2.0 * t + p) * 0.5 * 2 * np.pi * 2.0 for p in [0, np.pi/2, np.pi, 3*np.pi/2]]
        actions = [np.sin(t + p) * 0.3 for p in [0, 1, 2, 3]]
        steps.append(Step(
            observation=Observation(joint_positions=jp, joint_velocities=jv),
            action=Action(values=actions, control_mode=ControlMode.TORQUE),
            reward=1.0,
            done=False,
            timestamp=t,
        ))
    steps[-1].done = True

    metadata = EpisodeMetadata(
        robot=robot,
        terrain=TerrainSpec(terrain_type=TerrainType.FLAT),
        source=DataSource.SIMULATION,
        sampling_rate_hz=50.0,
    )
    return Episode(episode_id="test_sin_001", steps=steps, metadata=metadata)


class TestValidation:
    def test_valid_episode(self, sample_episode):
        result = validate_episode(sample_episode)
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    def test_detects_dimension_mismatch(self):
        robot = RobotSpec(
            robot_id="bad", robot_name="Bad",
            morphology=Morphology.OTHER, n_joints=4, n_actuators=4, mass_kg=1.0,
        )
        steps = [
            Step(
                observation=Observation(joint_positions=[0.1, 0.2], joint_velocities=[0.1, 0.2]),
                action=Action(values=[0.1, 0.2]),
                timestamp=0.0,
            ),
            Step(
                observation=Observation(joint_positions=[0.1, 0.2, 0.3], joint_velocities=[0.1, 0.2, 0.3]),
                action=Action(values=[0.1, 0.2]),
                timestamp=0.02,
            ),
        ]
        metadata = EpisodeMetadata(robot=robot)
        ep = Episode(episode_id="dim_err", steps=steps, metadata=metadata)
        result = validate_episode(ep)
        assert not result.valid

    def test_detects_non_monotonic_timestamps(self):
        robot = RobotSpec(
            robot_id="ts", robot_name="TS",
            morphology=Morphology.OTHER, n_joints=2, n_actuators=2, mass_kg=1.0,
        )
        steps = [
            Step(
                observation=Observation(joint_positions=[0.1, 0.2], joint_velocities=[0.1, 0.2]),
                action=Action(values=[0.1, 0.2]),
                timestamp=0.02,
            ),
            Step(
                observation=Observation(joint_positions=[0.1, 0.2], joint_velocities=[0.1, 0.2]),
                action=Action(values=[0.1, 0.2]),
                timestamp=0.01,  # goes backward!
            ),
        ]
        metadata = EpisodeMetadata(robot=robot)
        ep = Episode(episode_id="ts_err", steps=steps, metadata=metadata)
        result = validate_episode(ep)
        assert not result.valid


class TestMetrics:
    def test_compute_metrics(self, sample_episode):
        metrics = compute_metrics(sample_episode)
        assert metrics.n_steps == 100
        assert metrics.duration_seconds > 0
        assert metrics.total_energy_joules > 0
        assert metrics.action_magnitude_mean > 0

    def test_stride_frequency_detection(self, sample_episode):
        metrics = compute_metrics(sample_episode)
        # We created sinusoidal joints at 2 Hz, so stride freq should be near 2
        assert 1.0 < metrics.stride_frequency < 4.0, (
            f"Expected ~2 Hz stride freq, got {metrics.stride_frequency}"
        )

    def test_smoothness(self, sample_episode):
        metrics = compute_metrics(sample_episode)
        assert metrics.smoothness_index >= 0
