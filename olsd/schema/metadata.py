"""
OLSD Metadata Schema — Robot specifications, episode metadata, and dataset info.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from olsd.schema.trajectory import (
    ControlMode,
    DataSource,
    GaitType,
    Morphology,
    TerrainType,
)


# ---------------------------------------------------------------------------
# Robot specification
# ---------------------------------------------------------------------------


class JointInfo(BaseModel):
    """Per-joint specification."""

    name: str
    lower_limit: float  # rad
    upper_limit: float  # rad
    max_torque: float  # N·m
    max_velocity: float  # rad/s


class RobotSpec(BaseModel):
    """
    Hardware / model specification for a robot.
    This is shared across all episodes for the same robot.
    """

    robot_id: str  # Canonical ID, e.g., "unitree_go1"
    robot_name: str  # Human-readable, e.g., "Unitree Go1"
    morphology: Morphology
    n_joints: int
    n_actuators: int
    mass_kg: float
    joints: list[JointInfo] | None = None

    # Model file references (at least one required for sim)
    urdf_path: str | None = None
    mjcf_path: str | None = None

    # Physical dimensions (optional but useful)
    standing_height_m: float | None = None
    body_length_m: float | None = None
    n_legs: int | None = None
    dof_per_leg: int | None = None

    # Provenance
    manufacturer: str | None = None
    description: str | None = None


# ---------------------------------------------------------------------------
# Terrain specification
# ---------------------------------------------------------------------------


class TerrainSpec(BaseModel):
    """Parameters describing the terrain."""

    terrain_type: TerrainType
    friction_coefficient: float | None = None
    restitution: float | None = None
    slope_deg: float | None = None
    roughness: float | None = None  # 0.0 = smooth, 1.0 = very rough
    heightmap_path: str | None = None  # path to heightmap if procedural
    description: str | None = None


# ---------------------------------------------------------------------------
# Domain randomization config
# ---------------------------------------------------------------------------


class DomainRandomization(BaseModel):
    """Records what was randomized during trajectory generation."""

    friction_range: tuple[float, float] | None = None
    mass_scale_range: tuple[float, float] | None = None  # e.g., (0.9, 1.1)
    joint_stiffness_range: tuple[float, float] | None = None
    terrain_roughness_range: tuple[float, float] | None = None
    sensor_noise_std: float | None = None
    action_delay_range: tuple[float, float] | None = None  # seconds
    gravity_range: tuple[float, float] | None = None
    enabled: bool = False


# ---------------------------------------------------------------------------
# Episode metadata
# ---------------------------------------------------------------------------


class EpisodeMetadata(BaseModel):
    """
    Metadata attached to each Episode.

    Contains robot info, terrain, gait, and provenance.
    """

    # Robot
    robot: RobotSpec

    # Environment
    terrain: TerrainSpec = Field(
        default_factory=lambda: TerrainSpec(terrain_type=TerrainType.FLAT)
    )
    domain_randomization: DomainRandomization | None = None

    # Task
    gait_type: GaitType | None = None
    target_speed_mps: float | None = None
    target_heading_rad: float | None = None
    task_description: str | None = None

    # Outcome
    success: bool = True
    actual_speed_mps: float | None = None
    energy_cost_joules: float | None = None
    distance_traveled_m: float | None = None

    # Control
    control_mode: ControlMode = ControlMode.TORQUE
    action_frequency_hz: float = 50.0
    observation_frequency_hz: float = 200.0

    # Quality tier (for assembled datasets)
    quality_tier: str | None = None  # "random", "expert", "domain_random"

    # Source / provenance
    source: DataSource = DataSource.SIMULATION
    simulator: str | None = None  # "mujoco", "isaac_gym", etc.
    simulator_version: str | None = None
    policy_name: str | None = None  # e.g., "ppo_flat_v1"
    random_seed: int | None = None

    # Attribution
    contributor: str | None = None
    institution: str | None = None
    license: str = "CC-BY-4.0"
    citation: str | None = None
    paper_url: str | None = None

    # Timestamps
    date_collected: datetime | None = None
    sampling_rate_hz: float = 50.0
    duration_seconds: float | None = None


# ---------------------------------------------------------------------------
# Dataset-level metadata
# ---------------------------------------------------------------------------


class DatasetInfo(BaseModel):
    """Top-level metadata for an OLSD dataset release."""

    name: str = "olsd"
    version: str = "0.1.0"
    description: str = "Open Locomotion Skills Dataset"
    license: str = "CC-BY-4.0"
    total_episodes: int = 0
    total_steps: int = 0
    robots: list[str] = Field(default_factory=list)  # robot_ids
    terrains: list[str] = Field(default_factory=list)  # terrain types
    morphologies: list[str] = Field(default_factory=list)
    contributors: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    homepage: str = "https://github.com/open-locomotion-skills/olsd"
