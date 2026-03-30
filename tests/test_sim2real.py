"""Tests for OLSD sim-to-real modules."""

from __future__ import annotations

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
from olsd.sim2real.alignment_eval import (
    evaluate_alignment,
    evaluate_episode_alignment,
    save_alignment_report,
)
from olsd.sim2real.domain_config import (
    RobotSim2RealConfig,
    derive_domain_randomization,
    load_sim2real_config,
    save_sim2real_config,
)
from olsd.sim2real.go1_env import (
    GO1_POLICY_ACTION_RANGE,
    GO1_STANDING_POSE,
    Go1SimEnv,
    build_go1_xml,
)
from olsd.sim2real.system_id import (
    SimParams,
    TemplateReplayBackend,
    identify_params,
    simulate_episodes,
)
from olsd.sim2real.terrain import generate_terrain_xml


def _make_episode(
    n_steps: int = 50,
    n_joints: int = 12,
    robot_id: str = "unitree_go1",
    offset: float = 0.0,
) -> Episode:
    steps = []
    for idx in range(n_steps):
        t = idx * 0.02
        base = np.sin(np.linspace(0.0, np.pi, n_joints) + t) + offset
        velocity = np.cos(np.linspace(0.0, np.pi, n_joints) + t)
        steps.append(
            Step(
                observation=Observation(
                    joint_positions=base.tolist(),
                    joint_velocities=velocity.tolist(),
                ),
                action=Action(
                    values=(base * 0.5).tolist(),
                    control_mode=ControlMode.POSITION,
                ),
                reward=float(np.sum(base)),
                done=(idx == n_steps - 1),
                timestamp=t,
            )
        )

    robot = RobotSpec(
        robot_id=robot_id,
        robot_name=robot_id,
        morphology=Morphology.QUADRUPED,
        n_joints=n_joints,
        n_actuators=n_joints,
        mass_kg=12.0,
    )
    metadata = EpisodeMetadata(
        robot=robot,
        terrain=TerrainSpec(terrain_type=TerrainType.FLAT),
        source=DataSource.SIMULATION,
        sampling_rate_hz=50.0,
    )
    return Episode(episode_id=f"{robot_id}_{n_steps}_{offset}", steps=steps, metadata=metadata)


class TestSimParams:
    def test_default_values(self):
        params = SimParams.default()
        assert params.global_friction == pytest.approx(1.0)
        assert params.mass_scale == pytest.approx(1.0)

    def test_vector_roundtrip(self):
        params = SimParams.default()
        roundtrip = SimParams.from_vector(params.to_vector())
        assert roundtrip == params

    def test_dict_roundtrip(self):
        params = SimParams.default()
        roundtrip = SimParams.from_dict(params.to_dict())
        assert roundtrip == params

    def test_invalid_values_rejected(self):
        with pytest.raises(ValueError):
            SimParams(
                global_friction=-1.0,
                mass_scale=1.0,
                joint_damping_scale=1.0,
                joint_armature_scale=1.0,
                kp_scale=1.0,
                kd_scale=1.0,
                actuator_latency_ms=0.0,
                observation_noise_std=0.0,
            )


class TestDomainConfig:
    def test_derive_domain_randomization(self):
        config = derive_domain_randomization(SimParams.default(), relative_margin=0.2)
        assert config.friction_range[0] < 1.0 < config.friction_range[1]
        assert config.observation_noise_std >= 0.005

    def test_save_load_roundtrip(self):
        config = RobotSim2RealConfig(
            robot_id="unitree_go1",
            identified_params=SimParams.default(),
            domain_randomization=derive_domain_randomization(SimParams.default()),
            template_only=True,
            notes=["template only"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_sim2real_config(config, Path(tmpdir) / "go1.yaml")
            loaded = load_sim2real_config(path)
        assert loaded.robot_id == config.robot_id
        assert loaded.template_only is True
        assert loaded.identified_params == config.identified_params


class TestTerrainXml:
    def test_flat_xml(self):
        xml = generate_terrain_xml(TerrainType.FLAT)
        assert 'type="plane"' in xml
        assert 'name="ground"' in xml

    def test_slope_xml(self):
        xml = generate_terrain_xml(TerrainType.SLOPE, params={"angle_deg": 12})
        assert 'name="slope_ground"' in xml
        assert 'euler="0 12.0 0"' in xml

    def test_stairs_xml(self):
        xml = generate_terrain_xml(TerrainType.STAIRS, params={"n_steps": 4})
        assert 'name="stair_0"' in xml
        assert 'name="stair_3"' in xml

    def test_go1_xml_uses_terrain(self):
        xml = build_go1_xml(TerrainType.SLOPE, {"angle_deg": 18}, SimParams.default())
        assert "slope_ground" in xml
        assert 'kp="40.0"' in xml


class TestAlignmentEval:
    def test_identical_episode_alignment(self):
        episode = _make_episode()
        report = evaluate_episode_alignment(episode, episode)
        assert report["joint_rmse"] == pytest.approx(0.0)
        assert report["velocity_correlation"] == pytest.approx(1.0)

    def test_alignment_detects_shift(self):
        real_episode = _make_episode(offset=0.0)
        sim_episode = _make_episode(offset=0.2)
        report = evaluate_episode_alignment(real_episode, sim_episode)
        assert report["joint_rmse"] > 0.0

    def test_alignment_handles_shared_dimensions(self):
        real_episode = _make_episode(n_steps=60, n_joints=12)
        sim_episode = _make_episode(n_steps=45, n_joints=8)
        report = evaluate_episode_alignment(real_episode, sim_episode)
        assert report["shared_steps"] == 45
        assert report["shared_joints"] == 8

    def test_alignment_aggregate(self):
        real_episodes = [_make_episode(offset=0.0), _make_episode(offset=0.1)]
        sim_episodes = [_make_episode(offset=0.0), _make_episode(offset=0.15)]
        report = evaluate_alignment(real_episodes, sim_episodes)
        assert report["n_pairs"] == 2
        assert len(report["per_joint_rmse"]) == 12

    def test_alignment_report_save(self):
        report = evaluate_alignment([_make_episode()], [_make_episode(offset=0.1)])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_alignment_report(report, Path(tmpdir) / "report.json")
            assert path.exists()


class TestSystemId:
    def test_template_backend_marks_simulation(self):
        episode = _make_episode()
        backend = TemplateReplayBackend()
        simulated = backend.rollout_episode(
            reference_episode=episode,
            params=SimParams.default(),
            robot_config=episode.metadata.robot,
            mjcf_xml=None,
        )
        assert simulated.metadata.source == DataSource.SIMULATION
        assert simulated.metadata.dynamics_params is not None

    def test_simulate_episodes_count(self):
        episodes = [_make_episode(), _make_episode(offset=0.1)]
        simulated = simulate_episodes(
            real_trajectories=episodes,
            params=SimParams.default(),
            robot_config=episodes[0].metadata.robot,
            mjcf_xml=None,
            simulator_backend=TemplateReplayBackend(),
        )
        assert len(simulated) == len(episodes)

    def test_identify_params_improves_objective(self):
        target = SimParams(
            global_friction=1.2,
            mass_scale=0.95,
            joint_damping_scale=1.1,
            joint_armature_scale=0.9,
            kp_scale=1.05,
            kd_scale=0.85,
            actuator_latency_ms=4.0,
            observation_noise_std=0.02,
        )

        def objective_fn(candidate: SimParams) -> float:
            diff = candidate.to_vector() - target.to_vector()
            return float(np.dot(diff, diff))

        best = identify_params(
            real_trajectories=[_make_episode()],
            robot_config=_make_episode().metadata.robot,
            mjcf_xml=None,
            n_generations=6,
            population_size=10,
            objective_fn=objective_fn,
            simulator_backend=TemplateReplayBackend(),
            seed=123,
        )
        assert objective_fn(best) < objective_fn(SimParams.default())

    def test_identify_params_returns_simparams(self):
        best = identify_params(
            real_trajectories=[_make_episode()],
            robot_config=_make_episode().metadata.robot,
            mjcf_xml=None,
            n_generations=2,
            population_size=4,
            objective_fn=lambda params: float(np.sum(params.to_vector() ** 2)),
            simulator_backend=TemplateReplayBackend(),
            seed=0,
        )
        assert isinstance(best, SimParams)


class TestGo1EnvSmoke:
    def test_env_reset_step(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("mujoco")

        env = Go1SimEnv(max_steps=4)
        observation, _ = env.reset(seed=0)
        assert observation.shape == (35,)
        next_observation, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        assert next_observation.shape == (35,)
        assert isinstance(reward, float)
        assert "x_velocity" in info
        env.close()

    def test_zero_action_maps_to_standing_pose(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("mujoco")

        env = Go1SimEnv(max_steps=4)
        env.reset(seed=0)
        target = env.denormalize_action(np.zeros(12, dtype=np.float32))
        assert np.allclose(target, GO1_STANDING_POSE)
        max_target = env.denormalize_action(np.ones(12, dtype=np.float32))
        assert np.allclose(max_target - GO1_STANDING_POSE, GO1_POLICY_ACTION_RANGE)
        env.close()

    def test_survival_reward_beats_fall(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("mujoco")

        env = Go1SimEnv(max_steps=4)
        env.reset(seed=0)
        stable_reward = env._compute_reward(GO1_STANDING_POSE, terminated=False)
        env.data.qpos[2] = 0.05
        env.data.qpos[3] = 0.0
        fallen_reward = env._compute_reward(GO1_STANDING_POSE, terminated=True)
        assert stable_reward > fallen_reward
        env.close()

    def test_slope_reset_survives_first_step(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("mujoco")

        env = Go1SimEnv(
            terrain_type=TerrainType.SLOPE,
            terrain_params={"angle_deg": 5.0, "friction": 1.5},
            max_steps=40,
        )
        env.reset(seed=0)
        terminated = False
        truncated = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(np.zeros(12, dtype=np.float32))
            if terminated or truncated:
                break
        assert not terminated
        assert not truncated
        env.close()
