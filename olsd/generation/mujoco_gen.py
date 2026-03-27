"""
OLSD MuJoCo Trajectory Generator — Generate locomotion trajectories
using MuJoCo + Gymnasium environments.

Supports:
  - Standard MuJoCo envs (HalfCheetah, Ant, Walker2d, Hopper)
  - Custom policies (trained RL checkpoints or random)
  - Domain randomization
  - Configurable terrain and gait parameters
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import gymnasium as gym
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
from olsd.schema.metadata import DomainRandomization

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robot environment mapping
# ---------------------------------------------------------------------------

GYMNASIUM_ROBOTS: dict[str, dict[str, Any]] = {
    "halfcheetah": {
        "env_id": "HalfCheetah-v5",
        "morphology": Morphology.OTHER,
        "n_joints": 6,
        "n_actuators": 6,
        "mass_kg": 14.0,
        "obs_joint_slice": (0, 6),      # indices into obs for joint positions
        "obs_vel_slice": (6, 12),       # indices into obs for joint velocities
    },
    "ant": {
        "env_id": "Ant-v5",
        "morphology": Morphology.QUADRUPED,
        "n_joints": 8,
        "n_actuators": 8,
        "mass_kg": 13.5,
        "obs_joint_slice": (5, 13),     # skip com/quat, get joint angles
        "obs_vel_slice": (19, 27),      # joint velocities
    },
    "walker2d": {
        "env_id": "Walker2d-v5",
        "morphology": Morphology.BIPED,
        "n_joints": 6,
        "n_actuators": 6,
        "mass_kg": 11.0,
        "obs_joint_slice": (1, 7),
        "obs_vel_slice": (7, 13),
    },
    "hopper": {
        "env_id": "Hopper-v5",
        "morphology": Morphology.BIPED,
        "n_joints": 3,
        "n_actuators": 3,
        "mass_kg": 5.0,
        "obs_joint_slice": (1, 4),
        "obs_vel_slice": (4, 7),
    },
}


# ---------------------------------------------------------------------------
# Policy types
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Random action sampler."""

    def __init__(self, action_space: gym.spaces.Box):
        self.action_space = action_space

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


class SB3Policy:
    """Wrapper for Stable Baselines 3 trained policies."""

    def __init__(self, model_path: str | Path):
        from stable_baselines3 import PPO, SAC

        path = Path(model_path)
        # Try loading as PPO first, then SAC
        try:
            self.model = PPO.load(path)
        except Exception:
            self.model = SAC.load(path)
        logger.info(f"Loaded SB3 policy from {path}")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class ExpertHeuristicPolicy:
    """
    Simple heuristic controller for basic locomotion.
    Produces sinusoidal joint commands (walk-like pattern).
    """

    def __init__(
        self,
        n_actuators: int,
        frequency: float = 2.0,
        amplitude: float = 0.5,
        phase_offsets: list[float] | None = None,
    ):
        self.n_actuators = n_actuators
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase_offsets = phase_offsets or [
            i * 2 * np.pi / n_actuators for i in range(n_actuators)
        ]
        self._t = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        t = self._t * 0.02  # assume 50 Hz
        self._t += 1
        actions = np.array([
            self.amplitude * np.sin(2 * np.pi * self.frequency * t + phase)
            for phase in self.phase_offsets
        ])
        return actions


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_trajectories(
    robot_id: str,
    n_episodes: int = 100,
    max_steps: int = 1000,
    policy: Callable | str | None = None,
    terrain: TerrainType = TerrainType.FLAT,
    gait: GaitType | None = None,
    target_speed: float | None = None,
    domain_randomization: bool = False,
    seed: int = 42,
    configs_dir: str | Path = "configs/robots",
) -> list[Episode]:
    """
    Generate locomotion trajectories using Gymnasium MuJoCo envs.

    Args:
        robot_id: Robot identifier ("halfcheetah", "ant", "walker2d", "hopper")
        n_episodes: Number of episodes to generate
        max_steps: Maximum steps per episode
        policy: Action policy — "random", "heuristic", path to SB3 checkpoint, or callable
        terrain: Terrain type label
        gait: Gait type label
        target_speed: Target forward speed (m/s)
        domain_randomization: Whether to apply domain randomization
        seed: Random seed
        configs_dir: Directory containing robot YAML configs
    """
    if robot_id not in GYMNASIUM_ROBOTS:
        raise ValueError(
            f"Unknown robot: {robot_id}. "
            f"Available: {list(GYMNASIUM_ROBOTS.keys())}"
        )

    robot_info = GYMNASIUM_ROBOTS[robot_id]
    env_id = robot_info["env_id"]

    # Load robot spec from YAML if available
    try:
        from olsd.pipeline.ingest import load_robot_by_id
        robot_spec = load_robot_by_id(robot_id, configs_dir)
    except FileNotFoundError:
        robot_spec = RobotSpec(
            robot_id=robot_id,
            robot_name=robot_info["env_id"],
            morphology=robot_info["morphology"],
            n_joints=robot_info["n_joints"],
            n_actuators=robot_info["n_actuators"],
            mass_kg=robot_info["mass_kg"],
        )

    # Create environment
    env = gym.make(env_id)

    # Set up policy
    action_policy = _resolve_policy(policy, env.action_space, robot_info["n_actuators"])

    # Domain randomization config
    dr_config = None
    if domain_randomization:
        dr_config = DomainRandomization(
            friction_range=(0.5, 2.0),
            mass_scale_range=(0.9, 1.1),
            sensor_noise_std=0.01,
            enabled=True,
        )

    episodes = []
    rng = np.random.default_rng(seed)

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx
        obs, info = env.reset(seed=ep_seed)
        steps = []
        done = False
        truncated = False
        t = 0

        # Apply domain randomization (noise on observations)
        noise_std = dr_config.sensor_noise_std if dr_config else 0.0

        while not (done or truncated) and t < max_steps:
            action = action_policy(obs)

            next_obs, reward, done, truncated, step_info = env.step(action)

            # Extract structured observation
            jp_slice = robot_info["obs_joint_slice"]
            jv_slice = robot_info["obs_vel_slice"]

            joint_pos = obs[jp_slice[0]:jp_slice[1]].tolist()
            joint_vel = obs[jv_slice[0]:jv_slice[1]].tolist()

            # Add sensor noise if DR enabled
            if noise_std > 0:
                joint_pos = (np.array(joint_pos) + rng.normal(0, noise_std, len(joint_pos))).tolist()
                joint_vel = (np.array(joint_vel) + rng.normal(0, noise_std, len(joint_vel))).tolist()

            step = Step(
                observation=Observation(
                    joint_positions=joint_pos,
                    joint_velocities=joint_vel,
                ),
                action=Action(
                    values=action.tolist(),
                    control_mode=ControlMode.TORQUE,
                ),
                reward=float(reward),
                done=bool(done),
                truncated=bool(truncated),
                timestamp=t / 50.0,  # 50 Hz default
            )
            steps.append(step)
            obs = next_obs
            t += 1

        if not steps:
            continue

        # Compute actual speed from rewards/info if available
        actual_speed = None
        if "x_velocity" in (step_info or {}):
            actual_speed = float(step_info["x_velocity"])

        metadata = EpisodeMetadata(
            robot=robot_spec,
            terrain=TerrainSpec(terrain_type=terrain),
            gait_type=gait,
            target_speed_mps=target_speed,
            actual_speed_mps=actual_speed,
            success=not done,  # done typically means fallen
            source=DataSource.SIMULATION,
            simulator="mujoco",
            simulator_version="gymnasium",
            policy_name=_policy_name(policy),
            random_seed=ep_seed,
            sampling_rate_hz=50.0,
            domain_randomization=dr_config,
            control_mode=ControlMode.TORQUE,
        )

        episode = Episode(
            episode_id=f"{robot_id}_{ep_idx:06d}_{uuid4().hex[:8]}",
            steps=steps,
            metadata=metadata,
        )
        episodes.append(episode)

        if (ep_idx + 1) % max(1, n_episodes // 10) == 0:
            logger.info(f"Generated {ep_idx + 1}/{n_episodes} episodes for {robot_id}")

    env.close()
    logger.info(
        f"Generated {len(episodes)} episodes for {robot_id} "
        f"({sum(ep.n_steps for ep in episodes)} total steps)"
    )
    return episodes


# ---------------------------------------------------------------------------
# Multi-robot batch generation
# ---------------------------------------------------------------------------


def generate_dataset(
    robots: list[str] | None = None,
    episodes_per_robot: int = 100,
    quality_tiers: list[str] | None = None,
    terrains: list[TerrainType] | None = None,
    seed: int = 42,
    configs_dir: str | Path = "configs/robots",
) -> list[Episode]:
    """
    Generate a full OLSD dataset across multiple robots and configurations.

    Args:
        robots: List of robot IDs. Defaults to all available.
        episodes_per_robot: Episodes per (robot, quality, terrain) combo
        quality_tiers: Policy quality levels: "random", "heuristic", "expert"
        terrains: Terrain types to generate for
        seed: Base random seed
    """
    robots = robots or list(GYMNASIUM_ROBOTS.keys())
    quality_tiers = quality_tiers or ["random", "heuristic"]
    terrains = terrains or [TerrainType.FLAT]

    all_episodes = []
    combo_idx = 0

    for robot_id in robots:
        for quality in quality_tiers:
            for terrain in terrains:
                logger.info(
                    f"Generating: robot={robot_id}, quality={quality}, "
                    f"terrain={terrain.value}"
                )
                eps = generate_trajectories(
                    robot_id=robot_id,
                    n_episodes=episodes_per_robot,
                    policy=quality,
                    terrain=terrain,
                    seed=seed + combo_idx * 10000,
                    configs_dir=configs_dir,
                )
                all_episodes.extend(eps)
                combo_idx += 1

    logger.info(
        f"Generated complete dataset: {len(all_episodes)} episodes across "
        f"{len(robots)} robots, {len(quality_tiers)} quality tiers, "
        f"{len(terrains)} terrains"
    )
    return all_episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_policy(
    policy: Callable | str | None,
    action_space: gym.spaces.Box,
    n_actuators: int,
) -> Callable:
    """Resolve policy specification to a callable."""
    if policy is None or policy == "random":
        return RandomPolicy(action_space)
    elif policy == "heuristic":
        return ExpertHeuristicPolicy(n_actuators=n_actuators)
    elif isinstance(policy, str):
        # Assume it's a path to an SB3 checkpoint
        return SB3Policy(policy)
    elif callable(policy):
        return policy
    else:
        raise ValueError(f"Unknown policy type: {type(policy)}")


def _policy_name(policy: Callable | str | None) -> str:
    """Get a human-readable name for the policy."""
    if policy is None or policy == "random":
        return "random"
    elif policy == "heuristic":
        return "heuristic"
    elif isinstance(policy, str):
        return f"sb3:{Path(policy).stem}"
    else:
        return type(policy).__name__
