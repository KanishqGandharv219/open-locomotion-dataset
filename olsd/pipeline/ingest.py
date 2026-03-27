"""
OLSD Data Ingestion Pipeline — Convert diverse formats to OLSD Episodes.

Supported sources:
  - HDF5  (D4RL-style or custom)
  - NumPy  (.npz archives)
  - CSV   (tabular trajectory logs)
  - Gymnasium / D4RL  (live environment rollouts)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import h5py
import numpy as np
import yaml

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


# ---------------------------------------------------------------------------
# Robot config loader
# ---------------------------------------------------------------------------


def load_robot_config(config_path: str | Path) -> RobotSpec:
    """Load a robot specification from a YAML config file."""
    path = Path(config_path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RobotSpec(**raw)


def load_robot_by_id(robot_id: str, configs_dir: str | Path = "configs/robots") -> RobotSpec:
    """Load a robot config by its ID (filename without .yaml)."""
    path = Path(configs_dir) / f"{robot_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No robot config found: {path}")
    return load_robot_config(path)


# ---------------------------------------------------------------------------
# HDF5 ingestion (D4RL-style)
# ---------------------------------------------------------------------------


def from_hdf5(
    path: str | Path,
    robot: RobotSpec | None = None,
    episode_key: str | None = None,
    n_joints: int | None = None,
    terrain: TerrainType = TerrainType.FLAT,
    control_mode: ControlMode = ControlMode.TORQUE,
    sampling_rate_hz: float = 50.0,
) -> list[Episode]:
    """
    Import episodes from an HDF5 file.

    Supports D4RL-style layout with keys: observations, actions, rewards,
    terminals, timeouts. Splits into episodes on terminal/timeout flags.
    """
    path = Path(path)
    episodes: list[Episode] = []

    with h5py.File(path, "r") as f:
        if episode_key:
            # Hierarchical: each episode in its own group
            for key in sorted(f.keys()):
                if not key.startswith(episode_key):
                    continue
                grp = f[key]
                ep = _hdf5_group_to_episode(
                    grp, robot, n_joints, terrain, control_mode, sampling_rate_hz
                )
                episodes.append(ep)
        else:
            # Flat D4RL-style: split by terminal/timeout flags
            obs = np.array(f["observations"])
            acts = np.array(f["actions"])
            rewards = np.array(f.get("rewards", np.zeros(len(obs))))
            terminals = np.array(f.get("terminals", np.zeros(len(obs), dtype=bool)))
            timeouts = np.array(f.get("timeouts", np.zeros(len(obs), dtype=bool)))

            split_mask = np.logical_or(terminals, timeouts)
            split_indices = np.where(split_mask)[0] + 1

            ep_starts = np.concatenate([[0], split_indices])
            ep_ends = np.concatenate([split_indices, [len(obs)]])

            for start, end in zip(ep_starts, ep_ends):
                if start >= end:
                    continue
                ep_obs = obs[start:end]
                ep_acts = acts[start:end]
                ep_rews = rewards[start:end]
                ep_terms = terminals[start:end]

                ep = _arrays_to_episode(
                    observations=ep_obs,
                    actions=ep_acts,
                    rewards=ep_rews,
                    dones=ep_terms,
                    robot=robot,
                    n_joints=n_joints,
                    terrain=terrain,
                    control_mode=control_mode,
                    sampling_rate_hz=sampling_rate_hz,
                )
                episodes.append(ep)

    logger.info(f"Loaded {len(episodes)} episodes from {path}")
    return episodes


def _hdf5_group_to_episode(
    grp: h5py.Group,
    robot: RobotSpec | None,
    n_joints: int | None,
    terrain: TerrainType,
    control_mode: ControlMode,
    sampling_rate_hz: float,
) -> Episode:
    """Convert a single HDF5 group to an Episode."""
    obs = np.array(grp["observations"])
    acts = np.array(grp["actions"])
    rewards = np.array(grp.get("rewards", np.zeros(len(obs))))
    dones = np.array(grp.get("terminals", np.zeros(len(obs), dtype=bool)))
    return _arrays_to_episode(obs, acts, rewards, dones, robot, n_joints, terrain, control_mode, sampling_rate_hz)


# ---------------------------------------------------------------------------
# NumPy ingestion
# ---------------------------------------------------------------------------


def from_numpy(
    path: str | Path,
    robot: RobotSpec | None = None,
    n_joints: int | None = None,
    terrain: TerrainType = TerrainType.FLAT,
    control_mode: ControlMode = ControlMode.TORQUE,
    sampling_rate_hz: float = 50.0,
    obs_key: str = "observations",
    act_key: str = "actions",
    rew_key: str = "rewards",
    done_key: str = "dones",
) -> list[Episode]:
    """
    Import episodes from a .npz archive.

    Expects arrays named by the key arguments. If 'episode_starts' is present,
    uses it for splitting; otherwise treats as a single episode.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    obs = data[obs_key]
    acts = data[act_key]
    rewards = data.get(rew_key, np.zeros(len(obs)))
    dones = data.get(done_key, np.zeros(len(obs), dtype=bool))

    # Check for episode boundary info
    if "episode_starts" in data:
        starts = data["episode_starts"]
    else:
        # Split on done flags, or treat as single episode
        done_indices = np.where(dones)[0] + 1
        if len(done_indices) > 0:
            starts = np.concatenate([[0], done_indices])
        else:
            starts = np.array([0])

    ends = np.concatenate([starts[1:], [len(obs)]])

    episodes = []
    for s, e in zip(starts, ends):
        if s >= e:
            continue
        ep = _arrays_to_episode(
            obs[s:e], acts[s:e], rewards[s:e], dones[s:e],
            robot, n_joints, terrain, control_mode, sampling_rate_hz,
        )
        episodes.append(ep)

    logger.info(f"Loaded {len(episodes)} episodes from {path}")
    return episodes


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------


def from_csv(
    path: str | Path,
    column_mapping: dict[str, str | list[str]],
    robot: RobotSpec | None = None,
    n_joints: int | None = None,
    terrain: TerrainType = TerrainType.FLAT,
    control_mode: ControlMode = ControlMode.TORQUE,
    sampling_rate_hz: float = 50.0,
    episode_column: str | None = None,
) -> list[Episode]:
    """
    Import episodes from a CSV file.

    Args:
        column_mapping: Maps OLSD field names to CSV column names.
            Example: {
                "joint_positions": ["q0", "q1", "q2", ...],
                "joint_velocities": ["qd0", "qd1", "qd2", ...],
                "actions": ["a0", "a1", "a2", ...],
                "reward": "reward",
                "done": "terminal",
            }
        episode_column: If present, groups rows by this column into episodes.
    """
    import pandas as pd

    df = pd.read_csv(path)

    if episode_column and episode_column in df.columns:
        groups = df.groupby(episode_column)
    else:
        groups = [(0, df)]

    episodes = []
    for _, group_df in groups:
        n = len(group_df)
        if n == 0:
            continue

        # Extract joint positions
        jp_cols = column_mapping.get("joint_positions", [])
        if isinstance(jp_cols, str):
            jp_cols = [jp_cols]
        joint_pos = group_df[jp_cols].values if jp_cols else np.zeros((n, n_joints or 1))

        # Extract joint velocities
        jv_cols = column_mapping.get("joint_velocities", [])
        if isinstance(jv_cols, str):
            jv_cols = [jv_cols]
        joint_vel = group_df[jv_cols].values if jv_cols else np.zeros_like(joint_pos)

        # Actions
        a_cols = column_mapping.get("actions", [])
        if isinstance(a_cols, str):
            a_cols = [a_cols]
        actions = group_df[a_cols].values if a_cols else np.zeros_like(joint_pos)

        # Reward and done
        rew_col = column_mapping.get("reward", None)
        rewards = group_df[rew_col].values if rew_col and rew_col in group_df.columns else np.zeros(n)

        done_col = column_mapping.get("done", None)
        dones = group_df[done_col].values.astype(bool) if done_col and done_col in group_df.columns else np.zeros(n, dtype=bool)

        obs_array = np.concatenate([joint_pos, joint_vel], axis=1)
        ep = _arrays_to_episode(
            obs_array, actions, rewards, dones,
            robot, n_joints or joint_pos.shape[1], terrain, control_mode, sampling_rate_hz,
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
        )
        episodes.append(ep)

    logger.info(f"Loaded {len(episodes)} episodes from {path}")
    return episodes


# ---------------------------------------------------------------------------
# Gymnasium / D4RL live rollout
# ---------------------------------------------------------------------------


def from_gymnasium(
    env_id: str,
    n_episodes: int = 100,
    policy: Any | None = None,
    robot: RobotSpec | None = None,
    terrain: TerrainType = TerrainType.FLAT,
    max_steps: int = 1000,
    seed: int = 42,
) -> list[Episode]:
    """
    Collect episodes from a Gymnasium environment.

    Args:
        env_id: Gymnasium env ID (e.g., "HalfCheetah-v5", "Ant-v5")
        n_episodes: Number of episodes to collect
        policy: Callable (obs → action). If None, uses random actions.
        max_steps: Maximum steps per episode
        seed: Random seed
    """
    import gymnasium as gym

    env = gym.make(env_id)
    episodes = []

    for ep_idx in range(n_episodes):
        obs, info = env.reset(seed=seed + ep_idx)
        steps = []
        done = False
        truncated = False
        t = 0

        while not (done or truncated) and t < max_steps:
            if policy is not None:
                action = policy(obs)
            else:
                action = env.action_space.sample()

            next_obs, reward, done, truncated, info = env.step(action)

            n_act = len(action)
            obs_len = len(obs)
            # Heuristic: first half is positions, second half is velocities
            n_j = min(n_act, obs_len // 2) if obs_len > n_act else n_act

            step = Step(
                observation=Observation(
                    joint_positions=obs[:n_j].tolist(),
                    joint_velocities=obs[n_j:2 * n_j].tolist() if obs_len >= 2 * n_j else [0.0] * n_j,
                ),
                action=Action(values=action.tolist(), control_mode=ControlMode.TORQUE),
                reward=float(reward),
                done=bool(done),
                truncated=bool(truncated),
                timestamp=t / 50.0,
            )
            steps.append(step)
            obs = next_obs
            t += 1

        if len(steps) == 0:
            continue

        _robot = robot or RobotSpec(
            robot_id=env_id.lower().replace("-", "_"),
            robot_name=env_id,
            morphology=Morphology.OTHER,
            n_joints=n_j,
            n_actuators=n_act,
            mass_kg=0.0,
        )

        metadata = EpisodeMetadata(
            robot=_robot,
            terrain=TerrainSpec(terrain_type=terrain),
            success=not done,  # done usually means fallen
            source=DataSource.SIMULATION,
            simulator="mujoco",
            sampling_rate_hz=50.0,
        )

        episodes.append(Episode(
            episode_id=str(uuid4()),
            steps=steps,
            metadata=metadata,
        ))

    env.close()
    logger.info(f"Collected {len(episodes)} episodes from {env_id}")
    return episodes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _arrays_to_episode(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    robot: RobotSpec | None,
    n_joints: int | None,
    terrain: TerrainType,
    control_mode: ControlMode,
    sampling_rate_hz: float,
    joint_positions: np.ndarray | None = None,
    joint_velocities: np.ndarray | None = None,
) -> Episode:
    """Convert raw arrays into an OLSD Episode."""
    n_steps = len(observations)
    n_act = actions.shape[1] if actions.ndim > 1 else 1
    obs_dim = observations.shape[1] if observations.ndim > 1 else 1

    # If explicit joint arrays not provided, split observation heuristically
    if joint_positions is None:
        _nj = n_joints or min(n_act, obs_dim // 2)
        joint_positions = observations[:, :_nj] if obs_dim > _nj else observations
        joint_velocities = observations[:, _nj:2 * _nj] if obs_dim >= 2 * _nj else np.zeros_like(joint_positions)
    else:
        _nj = joint_positions.shape[1]

    steps = []
    dt = 1.0 / sampling_rate_hz

    for i in range(n_steps):
        step = Step(
            observation=Observation(
                joint_positions=joint_positions[i].tolist(),
                joint_velocities=joint_velocities[i].tolist(),
            ),
            action=Action(
                values=actions[i].tolist() if actions.ndim > 1 else [float(actions[i])],
                control_mode=control_mode,
            ),
            reward=float(rewards[i]),
            done=bool(dones[i]),
            timestamp=i * dt,
        )
        steps.append(step)

    _robot = robot or RobotSpec(
        robot_id="unknown",
        robot_name="Unknown Robot",
        morphology=Morphology.OTHER,
        n_joints=_nj,
        n_actuators=n_act,
        mass_kg=0.0,
    )

    metadata = EpisodeMetadata(
        robot=_robot,
        terrain=TerrainSpec(terrain_type=terrain),
        source=DataSource.SIMULATION,
        sampling_rate_hz=sampling_rate_hz,
    )

    return Episode(
        episode_id=str(uuid4()),
        steps=steps,
        metadata=metadata,
    )
