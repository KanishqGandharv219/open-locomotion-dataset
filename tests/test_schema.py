"""Tests for OLSD schema models."""

import pytest

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def robot():
    return RobotSpec(
        robot_id="test_robot",
        robot_name="Test Robot",
        morphology=Morphology.QUADRUPED,
        n_joints=8,
        n_actuators=8,
        mass_kg=10.0,
    )


@pytest.fixture
def observation():
    return Observation(
        joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        joint_velocities=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    )


@pytest.fixture
def action():
    return Action(values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


@pytest.fixture
def step(observation, action):
    return Step(observation=observation, action=action, reward=1.0, done=False, timestamp=0.0)


@pytest.fixture
def episode(step, robot):
    steps = [
        Step(
            observation=Observation(
                joint_positions=[float(i)] * 8,
                joint_velocities=[float(i) * 0.1] * 8,
            ),
            action=Action(values=[float(i) * 0.01] * 8),
            reward=float(i),
            done=(i == 9),
            timestamp=i * 0.02,
        )
        for i in range(10)
    ]
    metadata = EpisodeMetadata(
        robot=robot,
        terrain=TerrainSpec(terrain_type=TerrainType.FLAT),
        source=DataSource.SIMULATION,
    )
    return Episode(episode_id="test_ep_001", steps=steps, metadata=metadata)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestObservation:
    def test_valid(self, observation):
        assert len(observation.joint_positions) == 8
        assert len(observation.joint_velocities) == 8

    def test_empty_positions_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Observation(joint_positions=[], joint_velocities=[0.1])

    def test_empty_velocities_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Observation(joint_positions=[0.1], joint_velocities=[])

    def test_bad_quaternion_rejected(self):
        with pytest.raises(ValueError, match="4 elements"):
            Observation(
                joint_positions=[0.1],
                joint_velocities=[0.1],
                imu_orientation=[1.0, 0.0, 0.0],  # missing w
            )

    def test_optional_fields_default_none(self):
        obs = Observation(joint_positions=[0.1], joint_velocities=[0.2])
        assert obs.joint_torques is None
        assert obs.imu_orientation is None
        assert obs.contact_forces is None


class TestAction:
    def test_valid(self, action):
        assert len(action.values) == 8
        assert action.control_mode == ControlMode.TORQUE

    def test_empty_values_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Action(values=[])


class TestStep:
    def test_valid(self, step):
        assert step.reward == 1.0
        assert step.done is False
        assert step.timestamp == 0.0

    def test_defaults(self, observation, action):
        step = Step(observation=observation, action=action, timestamp=0.0)
        assert step.reward is None
        assert step.done is False
        assert step.truncated is False


class TestEpisode:
    def test_valid(self, episode):
        assert episode.n_steps == 10
        assert episode.episode_id == "test_ep_001"

    def test_empty_steps_rejected(self, robot):
        with pytest.raises(ValueError, match="at least one step"):
            Episode(
                episode_id="empty",
                steps=[],
                metadata=EpisodeMetadata(
                    robot=robot,
                    terrain=TerrainSpec(terrain_type=TerrainType.FLAT),
                ),
            )

    def test_duration(self, episode):
        assert episode.duration_seconds == pytest.approx(0.18, abs=0.01)

    def test_to_numpy(self, episode):
        data = episode.to_numpy()
        assert data["joint_positions"].shape == (10, 8)
        assert data["joint_velocities"].shape == (10, 8)
        assert data["actions"].shape == (10, 8)
        assert data["rewards"].shape == (10,)
        assert data["timestamps"].shape == (10,)


class TestRobotSpec:
    def test_valid(self, robot):
        assert robot.robot_id == "test_robot"
        assert robot.morphology == Morphology.QUADRUPED

    def test_morphology_enum(self):
        assert Morphology.QUADRUPED.value == "quadruped"
        assert Morphology.BIPED.value == "biped"


class TestEpisodeMetadata:
    def test_defaults(self, robot):
        meta = EpisodeMetadata(robot=robot)
        assert meta.terrain.terrain_type == TerrainType.FLAT
        assert meta.source == DataSource.SIMULATION
        assert meta.success is True
        assert meta.license == "CC-BY-4.0"
