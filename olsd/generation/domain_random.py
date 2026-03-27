"""
OLSD Domain Randomization — Configurable randomization for sim trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DomainRandomizationConfig:
    """
    Configuration for domain randomization during trajectory generation.

    Each parameter range defines [min, max] for uniform sampling.
    Set ranges to (1.0, 1.0) or (0.0, 0.0) to disable that axis.
    """

    # Physics
    friction_scale: tuple[float, float] = (0.7, 1.3)
    mass_scale: tuple[float, float] = (0.9, 1.1)
    joint_stiffness_scale: tuple[float, float] = (0.95, 1.05)
    joint_damping_scale: tuple[float, float] = (0.95, 1.05)
    gravity_scale: tuple[float, float] = (0.98, 1.02)  # fraction of 9.81

    # Sensor noise
    joint_position_noise_std: float = 0.01  # rad
    joint_velocity_noise_std: float = 0.05  # rad/s
    imu_orientation_noise_std: float = 0.005  # quaternion noise
    imu_accel_noise_std: float = 0.1  # m/s²

    # Action
    action_delay_steps: tuple[int, int] = (0, 2)  # steps of delay
    action_noise_std: float = 0.02

    # Terrain
    terrain_roughness: tuple[float, float] = (0.0, 0.05)  # meters
    terrain_slope_deg: tuple[float, float] = (0.0, 5.0)

    # Initial state
    initial_position_noise: float = 0.02  # m
    initial_velocity_noise: float = 0.05  # m/s

    enabled: bool = True

    def sample(self, rng: np.random.Generator | None = None) -> "SampledDomainParams":
        """Sample a concrete set of parameters from the ranges."""
        rng = rng or np.random.default_rng()

        return SampledDomainParams(
            friction_scale=rng.uniform(*self.friction_scale),
            mass_scale=rng.uniform(*self.mass_scale),
            joint_stiffness_scale=rng.uniform(*self.joint_stiffness_scale),
            joint_damping_scale=rng.uniform(*self.joint_damping_scale),
            gravity_scale=rng.uniform(*self.gravity_scale),
            action_delay_steps=rng.integers(*self.action_delay_steps, endpoint=True),
            terrain_roughness=rng.uniform(*self.terrain_roughness),
            terrain_slope_deg=rng.uniform(*self.terrain_slope_deg),
        )


@dataclass
class SampledDomainParams:
    """Concrete parameters sampled from a DomainRandomizationConfig."""

    friction_scale: float = 1.0
    mass_scale: float = 1.0
    joint_stiffness_scale: float = 1.0
    joint_damping_scale: float = 1.0
    gravity_scale: float = 1.0
    action_delay_steps: int = 0
    terrain_roughness: float = 0.0
    terrain_slope_deg: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "friction_scale": self.friction_scale,
            "mass_scale": self.mass_scale,
            "joint_stiffness_scale": self.joint_stiffness_scale,
            "joint_damping_scale": self.joint_damping_scale,
            "gravity_scale": self.gravity_scale,
            "action_delay_steps": self.action_delay_steps,
            "terrain_roughness": self.terrain_roughness,
            "terrain_slope_deg": self.terrain_slope_deg,
        }


def add_observation_noise(
    obs: dict[str, list[float]],
    config: DomainRandomizationConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, list[float]]:
    """Add sensor noise to an observation dict."""
    rng = rng or np.random.default_rng()
    noisy = dict(obs)

    if "joint_positions" in noisy and config.joint_position_noise_std > 0:
        jp = np.array(noisy["joint_positions"])
        jp += rng.normal(0, config.joint_position_noise_std, jp.shape)
        noisy["joint_positions"] = jp.tolist()

    if "joint_velocities" in noisy and config.joint_velocity_noise_std > 0:
        jv = np.array(noisy["joint_velocities"])
        jv += rng.normal(0, config.joint_velocity_noise_std, jv.shape)
        noisy["joint_velocities"] = jv.tolist()

    return noisy
