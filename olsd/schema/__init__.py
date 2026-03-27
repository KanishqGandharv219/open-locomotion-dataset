"""OLSD Schema — public API."""

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
    "get_reward",
]
