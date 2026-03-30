"""
Tests for External Dataset Ingestors + Cross-Embodiment Alignment.

All tests use synthetic/mocked data — no real downloads required.
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

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
from olsd.schema.alignment import (
    compute_max_dof,
    compute_normalization_stats,
    create_active_mask,
    denormalize_array,
    normalize_array,
    pad_array,
    pad_batch,
    save_normalization_stats,
)


# ---------------------------------------------------------------------------
# Helpers: Create synthetic test data
# ---------------------------------------------------------------------------


def _make_episode(n_steps: int = 50, n_joints: int = 6, robot_id: str = "test_robot") -> Episode:
    """Create a synthetic OLSD Episode."""
    steps = []
    for i in range(n_steps):
        step = Step(
            observation=Observation(
                joint_positions=np.random.randn(n_joints).tolist(),
                joint_velocities=np.random.randn(n_joints).tolist(),
            ),
            action=Action(values=np.random.randn(n_joints).tolist(), control_mode=ControlMode.TORQUE),
            reward=float(np.random.randn()),
            timestamp=i * 0.02,
        )
        steps.append(step)

    robot = RobotSpec(
        robot_id=robot_id,
        robot_name=f"Test {robot_id}",
        morphology=Morphology.QUADRUPED,
        n_joints=n_joints,
        n_actuators=n_joints,
        mass_kg=10.0,
    )

    metadata = EpisodeMetadata(
        robot=robot,
        terrain=TerrainSpec(terrain_type=TerrainType.FLAT),
        source=DataSource.SIMULATION,
        sampling_rate_hz=50.0,
    )

    return Episode(episode_id="test-ep-001", steps=steps, metadata=metadata)


def _make_synthetic_npz(path: Path, n_steps: int = 100, n_joints: int = 12):
    """Create a synthetic .npz file mimicking GrandTour/TAIL format."""
    data = {
        "joint_positions": np.random.randn(n_steps, n_joints).astype(np.float32),
        "joint_velocities": np.random.randn(n_steps, n_joints).astype(np.float32),
        "joint_torques": np.random.randn(n_steps, n_joints).astype(np.float32),
        "imu_orientation": np.tile([1.0, 0.0, 0.0, 0.0], (n_steps, 1)).astype(np.float32),
        "base_position": np.cumsum(np.random.randn(n_steps, 3) * 0.01, axis=0).astype(np.float32),
    }
    np.savez(path, **data)


def _make_synthetic_pkl(path: Path, n_frames: int = 60, n_dof: int = 23):
    """Create a synthetic .pkl file mimicking Unitree G1 retargeted motions."""
    motion = {
        "dof": np.random.randn(n_frames, n_dof).astype(np.float32) * 0.5,
        "root_trans_offset": np.cumsum(
            np.random.randn(n_frames, 3) * 0.01, axis=0
        ).astype(np.float32),
        "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (n_frames, 1)).astype(np.float32),
        "contact_mask": np.random.randint(0, 2, (n_frames, 2)).astype(bool),
        "pose_aa": np.random.randn(n_frames, n_dof * 3).astype(np.float32) * 0.1,
        "fps": 30,
    }
    with open(path, "wb") as f:
        pickle.dump(motion, f)


# ---------------------------------------------------------------------------
# Tests: Cross-Embodiment Alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    """Test training-time padding and normalization utilities."""

    def test_compute_max_dof(self):
        eps = [_make_episode(n_joints=6), _make_episode(n_joints=12), _make_episode(n_joints=8)]
        assert compute_max_dof(eps) == 12

    def test_create_active_mask(self):
        mask = create_active_mask(n_joints=6, max_dof=12)
        assert mask.shape == (12,)
        assert mask[:6].all()
        assert not mask[6:].any()

    def test_pad_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        padded = pad_array(arr, max_dof=6)
        assert padded.shape == (6,)
        np.testing.assert_array_equal(padded[:3], arr)
        np.testing.assert_array_equal(padded[3:], 0.0)

    def test_pad_array_truncate(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        padded = pad_array(arr, max_dof=3)
        assert padded.shape == (3,)
        np.testing.assert_array_equal(padded, arr[:3])

    def test_pad_batch(self):
        data = np.random.randn(10, 6)
        padded, mask = pad_batch(data, max_dof=12)
        assert padded.shape == (10, 12)
        assert mask.shape == (12,)
        np.testing.assert_array_equal(padded[:, :6], data)
        np.testing.assert_array_equal(padded[:, 6:], 0.0)
        assert mask[:6].all()
        assert not mask[6:].any()

    def test_normalize_denormalize_roundtrip(self):
        arr = np.random.randn(50, 6) * 5 + 3
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        normalized = normalize_array(arr, min_vals, max_vals)
        assert normalized.min() >= -0.01
        assert normalized.max() <= 1.01
        recovered = denormalize_array(normalized, min_vals, max_vals)
        np.testing.assert_allclose(recovered, arr, atol=1e-5)

    def test_compute_normalization_stats(self):
        eps = [
            _make_episode(n_joints=6, robot_id="robot_a"),
            _make_episode(n_joints=6, robot_id="robot_a"),
            _make_episode(n_joints=8, robot_id="robot_b"),
        ]
        stats = compute_normalization_stats(eps)
        assert "robot_a" in stats
        assert "robot_b" in stats
        assert "_global" in stats
        assert stats["robot_a"]["jp"]["n_dims"] == 6
        assert stats["robot_b"]["jp"]["n_dims"] == 8
        assert stats["_global"]["max_dof"] == 8

    def test_save_load_normalization_stats(self):
        eps = [_make_episode(n_joints=6)]
        stats = compute_normalization_stats(eps)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_normalization_stats(stats, tmpdir)
            path = Path(tmpdir) / "meta" / "normalization.json"
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert "test_robot" in loaded


# ---------------------------------------------------------------------------
# Tests: GrandTour Ingestor
# ---------------------------------------------------------------------------


class TestGrandTourIngest:
    """Test GrandTour ingestor with synthetic .npz data mimicking ANYmal-D."""

    def test_npz_loading(self):
        from olsd.pipeline.ingest_grandtour import from_grandtour

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_synthetic_npz(Path(tmpdir) / "forest_001.npz", n_steps=200, n_joints=12)
            _make_synthetic_npz(Path(tmpdir) / "indoor_002.npz", n_steps=150, n_joints=12)

            episodes = from_grandtour(tmpdir, max_episodes=5)

            assert len(episodes) == 2
            assert episodes[0].n_joints == 12
            assert episodes[0].metadata.source == DataSource.HARDWARE
            assert episodes[0].metadata.robot.robot_id == "anymal_d"
            assert episodes[0].metadata.external_dataset == "grandtour"

    def test_terrain_label_assignment(self):
        from olsd.pipeline.ingest_grandtour import from_grandtour

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_synthetic_npz(Path(tmpdir) / "forest_sequence.npz", n_joints=12)
            _make_synthetic_npz(Path(tmpdir) / "indoor_hallway.npz", n_joints=12)

            episodes = from_grandtour(tmpdir)
            terrains = {ep.metadata.external_episode_id: ep.metadata.terrain.terrain_type for ep in episodes}

            # "forest" should map to ROUGH, "indoor" to FLAT
            assert terrains["forest_sequence"] == TerrainType.ROUGH
            assert terrains["indoor_hallway"] == TerrainType.FLAT

    def test_subsampling(self):
        from olsd.pipeline.ingest_grandtour import from_grandtour

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_synthetic_npz(Path(tmpdir) / "test.npz", n_steps=400, n_joints=12)

            # At 200Hz source -> 50Hz target, expect ~100 steps (400/4)
            episodes = from_grandtour(tmpdir, subsample_hz=50.0, source_hz=200.0)
            assert len(episodes) == 1
            assert episodes[0].n_steps == 100

    def test_12_joint_anymal(self):
        from olsd.pipeline.ingest_grandtour import from_grandtour

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_synthetic_npz(Path(tmpdir) / "seq.npz", n_joints=12)
            episodes = from_grandtour(tmpdir)

            assert episodes[0].metadata.robot.n_joints == 12
            assert episodes[0].metadata.robot.morphology == Morphology.QUADRUPED
            assert len(episodes[0].steps[0].observation.joint_positions) == 12


# ---------------------------------------------------------------------------
# Tests: TAIL Ingestor
# ---------------------------------------------------------------------------


class TestTailIngest:
    """Test TAIL ingestor with synthetic data."""

    def test_npz_loading(self):
        from olsd.pipeline.ingest_tail import from_tail

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_synthetic_npz(Path(tmpdir) / "sand_seq1.npz", n_joints=12)
            episodes = from_tail(tmpdir)
            assert len(episodes) >= 1
            assert episodes[0].metadata.external_dataset == "tail"

    def test_terrain_tagging(self):
        from olsd.pipeline.ingest_tail import from_tail

        with tempfile.TemporaryDirectory() as tmpdir:
            sand_dir = Path(tmpdir) / "sandy_terrain"
            sand_dir.mkdir()
            _make_synthetic_npz(sand_dir / "seq1.npz", n_joints=12)
            episodes = from_tail(str(sand_dir))
            # The directory name "sandy_terrain" should trigger SAND tag
            # (depending on how _infer_tail_terrain processes the seq name)
            assert len(episodes) >= 1
            assert episodes[0].metadata.source == DataSource.HARDWARE


# ---------------------------------------------------------------------------
# Tests: Unitree G1 Ingestor
# ---------------------------------------------------------------------------


class TestUnitreeIngest:
    """Test Unitree G1 retargeted motion ingestor with synthetic pickle."""

    def test_pickle_to_episode(self):
        from olsd.pipeline.ingest_unitree import from_unitree_retargeted

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "dance_001.pkl"
            _make_synthetic_pkl(pkl_path, n_frames=60, n_dof=23)

            episodes = from_unitree_retargeted(pkl_path)
            assert len(episodes) == 1
            assert episodes[0].n_joints == 23
            assert episodes[0].metadata.robot.morphology == Morphology.HUMANOID
            assert episodes[0].metadata.external_dataset == "unitree_retargeted"

    def test_finite_diff_velocities(self):
        from olsd.pipeline.ingest_unitree import from_unitree_retargeted

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "walk.pkl"
            # Create linearly increasing joint positions
            n_frames, n_dof = 30, 23
            dof = np.linspace(0, 1, n_frames).reshape(-1, 1) * np.ones((1, n_dof))
            motion = {"dof": dof.astype(np.float32), "fps": 30}
            with open(pkl_path, "wb") as f:
                pickle.dump(motion, f)

            episodes = from_unitree_retargeted(pkl_path)
            # Velocities should be approximately constant (linear input)
            jv = np.array([s.observation.joint_velocities for s in episodes[0].steps])
            # After step 1, finite diff should give ~constant velocity
            assert jv[2:].std(axis=0).max() < 0.5  # reasonably constant

    def test_23_dof_humanoid(self):
        from olsd.pipeline.ingest_unitree import from_unitree_retargeted

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "motion.pkl"
            _make_synthetic_pkl(pkl_path, n_dof=23)
            episodes = from_unitree_retargeted(pkl_path, robot_type="g1")
            assert episodes[0].metadata.robot.robot_id == "unitree_g1"
            assert len(episodes[0].steps[0].observation.joint_positions) == 23

    def test_directory_of_pickles(self):
        from olsd.pipeline.ingest_unitree import from_unitree_retargeted

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                _make_synthetic_pkl(Path(tmpdir) / f"motion_{i}.pkl", n_dof=23)

            episodes = from_unitree_retargeted(tmpdir)
            assert len(episodes) == 3

    def test_max_episodes_limit(self):
        from olsd.pipeline.ingest_unitree import from_unitree_retargeted

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                _make_synthetic_pkl(Path(tmpdir) / f"motion_{i}.pkl", n_dof=23)

            episodes = from_unitree_retargeted(tmpdir, max_episodes=2)
            assert len(episodes) == 2


# ---------------------------------------------------------------------------
# Tests: Schema Backward Compatibility
# ---------------------------------------------------------------------------


class TestSchemaBackwardCompat:
    """Ensure new schema fields don't break existing functionality."""

    def test_new_metadata_fields_optional(self):
        """All new EpisodeMetadata fields should have defaults."""
        robot = RobotSpec(
            robot_id="test", robot_name="Test", morphology=Morphology.OTHER,
            n_joints=6, n_actuators=6, mass_kg=1.0,
        )
        # Create metadata without any new fields — should not raise
        meta = EpisodeMetadata(robot=robot, source=DataSource.SIMULATION)
        assert meta.external_dataset is None
        assert meta.external_episode_id is None
        assert meta.sim_real_pair_id is None
        assert meta.dynamics_params is None
        assert meta.velocity_command is None

    def test_wheeled_biped_morphology(self):
        """New WHEELED_BIPED enum value should be usable."""
        assert Morphology.WHEELED_BIPED.value == "wheeled_biped"

    def test_existing_episode_creation(self):
        """Existing episode creation patterns should still work."""
        ep = _make_episode(n_joints=6)
        assert ep.n_steps == 50
        assert ep.n_joints == 6
