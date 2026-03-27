"""
OLSD Trajectory Schema — Core data models for locomotion trajectories.

Defines the canonical Episode → Step structure using Pydantic v2.
Compatible with LeRobot v3 (Arrow/Parquet) export and RLDS conversion.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ControlMode(str, Enum):
    """Motor command type."""

    TORQUE = "torque"
    POSITION = "position"
    VELOCITY = "velocity"
    POSITION_VELOCITY = "position_velocity"


class Morphology(str, Enum):
    """Broad robot body plan."""

    QUADRUPED = "quadruped"
    BIPED = "biped"
    HUMANOID = "humanoid"
    HEXAPOD = "hexapod"
    OTHER = "other"


class TerrainType(str, Enum):
    """Surface type the robot is walking on."""

    FLAT = "flat"
    CONCRETE = "concrete"
    SAND = "sand"
    GRASS = "grass"
    GRAVEL = "gravel"
    ROCKY = "rocky"
    MUD = "mud"
    ICE = "ice"
    SNOW = "snow"
    SLOPE = "slope"
    STAIRS = "stairs"
    ROUGH = "rough"
    CUSTOM = "custom"


class GaitType(str, Enum):
    """Locomotion gait pattern."""

    WALK = "walk"
    TROT = "trot"
    PACE = "pace"
    GALLOP = "gallop"
    BOUND = "bound"
    CRAWL = "crawl"
    RUN = "run"
    PRONK = "pronk"
    OTHER = "other"


class DataSource(str, Enum):
    """Origin of the trajectory data."""

    SIMULATION = "simulation"
    HARDWARE = "hardware"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Core step-level models
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """
    Snapshot of robot state at a single timestep.

    All arrays are stored as plain Python lists for Pydantic compat.
    Conversion to numpy is done at the pipeline/SDK level.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required proprioception
    joint_positions: list[float]  # rad, shape [n_joints]
    joint_velocities: list[float]  # rad/s, shape [n_joints]

    # Optional proprioception
    joint_torques: list[float] | None = None  # N·m, shape [n_joints]
    joint_accelerations: list[float] | None = None  # rad/s², shape [n_joints]

    # IMU (required for real & sim with IMU)
    imu_orientation: list[float] | None = None  # quaternion [w, x, y, z]
    imu_angular_velocity: list[float] | None = None  # rad/s [3]
    imu_linear_acceleration: list[float] | None = None  # m/s² [3]

    # Contact
    contact_forces: list[float] | None = None  # N, shape [n_contacts × 3]
    contact_binary: list[bool] | None = None  # per-foot contact flag

    # Body state (often from sim; optional for hardware)
    base_position: list[float] | None = None  # [x, y, z] in world frame
    base_velocity: list[float] | None = None  # [vx, vy, vz] in world frame
    base_angular_velocity: list[float] | None = None  # [wx, wy, wz]

    # Vision (optional)
    rgb_path: str | None = None  # path to image frame
    depth_path: str | None = None  # path to depth frame

    @field_validator("joint_positions", "joint_velocities")
    @classmethod
    def _check_nonempty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("joint arrays must be non-empty")
        return v

    @field_validator("imu_orientation")
    @classmethod
    def _check_quaternion(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != 4:
            raise ValueError("imu_orientation must have exactly 4 elements (quaternion)")
        return v


class Action(BaseModel):
    """Motor command issued at a single timestep."""

    values: list[float]  # shape [n_actuators]
    control_mode: ControlMode = ControlMode.TORQUE

    @field_validator("values")
    @classmethod
    def _check_nonempty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("action values must be non-empty")
        return v


class Step(BaseModel):
    """Single timestep within an episode."""

    observation: Observation
    action: Action
    reward: float | None = None
    done: bool = False
    truncated: bool = False
    timestamp: float  # seconds from episode start
    info: dict[str, Any] | None = None  # extra per-step data


# ---------------------------------------------------------------------------
# Episode-level model
# ---------------------------------------------------------------------------


class Episode(BaseModel):
    """
    A complete locomotion trajectory: sequence of Steps + metadata.

    This is the atomic unit of the OLSD dataset.
    """

    episode_id: str  # unique identifier (UUID or contributor-assigned)
    steps: list[Step]
    metadata: "EpisodeMetadata"  # forward ref resolved at module load

    @field_validator("steps")
    @classmethod
    def _check_nonempty(cls, v: list[Step]) -> list[Step]:
        if len(v) == 0:
            raise ValueError("episode must have at least one step")
        return v

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def duration_seconds(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        return self.steps[-1].timestamp - self.steps[0].timestamp

    @property
    def n_joints(self) -> int:
        return len(self.steps[0].observation.joint_positions)

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert episode to dict of numpy arrays (for efficient processing)."""
        obs = self.steps
        return {
            "joint_positions": np.array([s.observation.joint_positions for s in obs]),
            "joint_velocities": np.array([s.observation.joint_velocities for s in obs]),
            "actions": np.array([s.action.values for s in obs]),
            "rewards": np.array([s.reward or 0.0 for s in obs]),
            "timestamps": np.array([s.timestamp for s in obs]),
            "dones": np.array([s.done for s in obs]),
        }


# Avoid circular import — metadata is defined in metadata.py
# but Episode references it. We use update_forward_refs at package init.
