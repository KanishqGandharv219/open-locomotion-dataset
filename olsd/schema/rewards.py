"""
OLSD Standardized Reward Functions for locomotion tasks.

Each reward is a callable class with configurable weights,
designed to be used both for training and for evaluating existing trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Base reward
# ---------------------------------------------------------------------------


class RewardFunction:
    """Base class for OLSD reward functions."""

    name: str = "base"
    description: str = ""

    def __call__(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: dict[str, np.ndarray],
        info: dict | None = None,
    ) -> float:
        raise NotImplementedError

    def compute_components(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: dict[str, np.ndarray],
        info: dict | None = None,
    ) -> dict[str, float]:
        """Return individual reward components for analysis."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Walking reward
# ---------------------------------------------------------------------------


@dataclass
class WalkingRewardWeights:
    """Tunable weights for the walking reward function."""

    forward_velocity: float = 1.0
    lateral_velocity_penalty: float = -0.5
    height_stability: float = -2.0
    orientation_penalty: float = -1.0
    energy_cost: float = -0.005
    action_smoothness: float = -0.01
    alive_bonus: float = 0.2


class WalkingReward(RewardFunction):
    """
    Standard walking reward function.

    r = w1 * exp(-||v - v_target||²)
      + w2 * |v_lateral|²
      + w3 * (z - z_nominal)²
      + w4 * ||orientation_error||²
      + w5 * ||torque||²
      + w6 * ||action_t - action_{t-1}||²
      + w7 * alive_bonus
    """

    name = "walking"
    description = "Reward for stable forward locomotion at a target velocity"

    def __init__(
        self,
        target_velocity: float = 1.0,
        nominal_height: float = 0.3,
        weights: WalkingRewardWeights | None = None,
    ):
        self.target_velocity = target_velocity
        self.nominal_height = nominal_height
        self.w = weights or WalkingRewardWeights()

    def compute_components(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: dict[str, np.ndarray],
        info: dict | None = None,
    ) -> dict[str, float]:
        info = info or {}

        # Forward velocity tracking (Gaussian kernel)
        v_forward = info.get("forward_velocity", 0.0)
        vel_error = (v_forward - self.target_velocity) ** 2
        r_velocity = float(np.exp(-vel_error))

        # Lateral velocity penalty
        v_lateral = info.get("lateral_velocity", 0.0)
        r_lateral = float(v_lateral**2)

        # Height stability
        height = info.get("base_height", self.nominal_height)
        r_height = float((height - self.nominal_height) ** 2)

        # Orientation (roll + pitch deviation from upright)
        orientation_error = info.get("orientation_error", 0.0)
        r_orient = float(orientation_error**2)

        # Energy cost (sum of squared torques)
        torques = obs.get("joint_torques", action)
        if torques is not None:
            r_energy = float(np.sum(np.square(torques)))
        else:
            r_energy = float(np.sum(np.square(action)))

        # Action smoothness (requires previous action in info)
        prev_action = info.get("prev_action", action)
        r_smooth = float(np.sum(np.square(action - prev_action)))

        # Alive bonus (not fallen)
        r_alive = 1.0 if not info.get("fallen", False) else 0.0

        return {
            "forward_velocity": r_velocity,
            "lateral_velocity_penalty": r_lateral,
            "height_stability": r_height,
            "orientation_penalty": r_orient,
            "energy_cost": r_energy,
            "action_smoothness": r_smooth,
            "alive_bonus": r_alive,
        }

    def __call__(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: dict[str, np.ndarray],
        info: dict | None = None,
    ) -> float:
        components = self.compute_components(obs, action, next_obs, info)
        w = self.w
        total = (
            w.forward_velocity * components["forward_velocity"]
            + w.lateral_velocity_penalty * components["lateral_velocity_penalty"]
            + w.height_stability * components["height_stability"]
            + w.orientation_penalty * components["orientation_penalty"]
            + w.energy_cost * components["energy_cost"]
            + w.action_smoothness * components["action_smoothness"]
            + w.alive_bonus * components["alive_bonus"]
        )
        return float(total)


# ---------------------------------------------------------------------------
# Terrain traversal reward
# ---------------------------------------------------------------------------


@dataclass
class TerrainRewardWeights:
    """Tunable weights for terrain traversal reward."""

    progress: float = 2.0
    balance: float = -1.0
    heading: float = -0.5
    energy_cost: float = -0.005
    contact_penalty: float = -1.0
    alive_bonus: float = 0.5


class TerrainTraversalReward(RewardFunction):
    """
    Reward for traversing uneven terrain.

    Emphasizes progress toward goal while maintaining balance.
    """

    name = "terrain_traversal"
    description = "Reward for traversing uneven/difficult terrain toward a goal"

    def __init__(
        self,
        target_heading: float = 0.0,
        weights: TerrainRewardWeights | None = None,
    ):
        self.target_heading = target_heading
        self.w = weights or TerrainRewardWeights()

    def compute_components(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: dict[str, np.ndarray],
        info: dict | None = None,
    ) -> dict[str, float]:
        info = info or {}

        r_progress = float(info.get("distance_delta", 0.0))
        r_balance = float(info.get("lateral_deviation", 0.0) ** 2)
        heading_err = info.get("heading_error", 0.0)
        r_heading = float(heading_err**2)

        torques = obs.get("joint_torques", action)
        r_energy = float(np.sum(np.square(torques if torques is not None else action)))

        contact_loss = float(info.get("contact_loss_count", 0))
        r_contact = contact_loss

        r_alive = 1.0 if not info.get("fallen", False) else 0.0

        return {
            "progress": r_progress,
            "balance": r_balance,
            "heading": r_heading,
            "energy_cost": r_energy,
            "contact_penalty": r_contact,
            "alive_bonus": r_alive,
        }

    def __call__(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: dict[str, np.ndarray],
        info: dict | None = None,
    ) -> float:
        c = self.compute_components(obs, action, next_obs, info)
        w = self.w
        return float(
            w.progress * c["progress"]
            + w.balance * c["balance"]
            + w.heading * c["heading"]
            + w.energy_cost * c["energy_cost"]
            + w.contact_penalty * c["contact_penalty"]
            + w.alive_bonus * c["alive_bonus"]
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REWARD_REGISTRY: dict[str, type[RewardFunction]] = {
    "walking": WalkingReward,
    "terrain_traversal": TerrainTraversalReward,
}


def get_reward(name: str, **kwargs) -> RewardFunction:
    """Instantiate a reward function by name."""
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward: {name}. Available: {list(REWARD_REGISTRY.keys())}")
    return REWARD_REGISTRY[name](**kwargs)
