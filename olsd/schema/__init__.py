"""OLSD Schema — public API."""

from olsd.schema.alignment import (
    compute_max_dof,
    compute_normalization_stats,
    create_active_mask,
    denormalize_array,
    load_normalization_stats,
    normalize_array,
    pad_array,
    pad_batch,
    save_normalization_stats,
)
from olsd.schema.metadata import (
    DatasetInfo,
    DomainRandomization,
    EpisodeMetadata,
    JointInfo,
    RobotSpec,
    TerrainSpec,
)
from olsd.schema.rewards import (
    RewardFunction,
    TerrainTraversalReward,
    WalkingReward,
    get_reward,
)
from olsd.schema.trajectory import (
    Action,
    ControlMode,
    DataSource,
    Episode,
    GaitType,
    Morphology,
    Observation,
    Step,
    TerrainType,
)

# Resolve forward references
Episode.model_rebuild()

__all__ = [
    "Action",
    "ControlMode",
    "DatasetInfo",
    "DataSource",
    "DomainRandomization",
    "Episode",
    "EpisodeMetadata",
    "GaitType",
    "JointInfo",
    "Morphology",
    "Observation",
    "RewardFunction",
    "RobotSpec",
    "Step",
    "TerrainSpec",
    "TerrainTraversalReward",
    "TerrainType",
    "WalkingReward",
    "compute_max_dof",
    "compute_normalization_stats",
    "create_active_mask",
    "denormalize_array",
    "get_reward",
    "load_normalization_stats",
    "normalize_array",
    "pad_array",
    "pad_batch",
    "save_normalization_stats",
]

