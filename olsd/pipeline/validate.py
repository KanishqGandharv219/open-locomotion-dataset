"""
OLSD Validation Pipeline — Schema compliance, range checks, and outlier detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from olsd.schema import Episode, RobotSpec

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating an episode or dataset."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    episode_id: str = ""

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        status = "✓ VALID" if self.valid else "✗ INVALID"
        parts = [f"[{status}] Episode: {self.episode_id}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN:  {w}")
        return "\n".join(parts)


def validate_episode(episode: Episode) -> ValidationResult:
    """Run all validation checks on a single episode."""
    result = ValidationResult(episode_id=episode.episode_id)

    _check_schema_completeness(episode, result)
    _check_array_consistency(episode, result)
    _check_physical_plausibility(episode, result)
    _check_temporal_consistency(episode, result)

    return result


def validate_dataset(episodes: list[Episode]) -> list[ValidationResult]:
    """Validate a list of episodes, return results for each."""
    results = []
    for ep in episodes:
        r = validate_episode(ep)
        results.append(r)
        if not r.valid:
            logger.warning(str(r))
    n_valid = sum(1 for r in results if r.valid)
    logger.info(f"Validation: {n_valid}/{len(results)} episodes valid")
    return results


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_schema_completeness(episode: Episode, result: ValidationResult) -> None:
    """Verify required fields are present and non-empty."""
    if not episode.steps:
        result.add_error("Episode has no steps")
        return

    step0 = episode.steps[0]

    if not step0.observation.joint_positions:
        result.add_error("First step has empty joint_positions")
    if not step0.observation.joint_velocities:
        result.add_error("First step has empty joint_velocities")
    if not step0.action.values:
        result.add_error("First step has empty action values")

    if not episode.metadata.robot.robot_id:
        result.add_error("Missing robot_id in metadata")

    # Check all steps have consistent dimensions
    n_jp = len(step0.observation.joint_positions)
    n_jv = len(step0.observation.joint_velocities)
    n_act = len(step0.action.values)

    for i, step in enumerate(episode.steps):
        if len(step.observation.joint_positions) != n_jp:
            result.add_error(f"Step {i}: joint_positions dim {len(step.observation.joint_positions)} != {n_jp}")
            break
        if len(step.observation.joint_velocities) != n_jv:
            result.add_error(f"Step {i}: joint_velocities dim {len(step.observation.joint_velocities)} != {n_jv}")
            break
        if len(step.action.values) != n_act:
            result.add_error(f"Step {i}: action dim {len(step.action.values)} != {n_act}")
            break


def _check_array_consistency(episode: Episode, result: ValidationResult) -> None:
    """Check that array sizes match robot spec."""
    robot = episode.metadata.robot

    if not episode.steps:
        return

    n_jp = len(episode.steps[0].observation.joint_positions)
    n_act = len(episode.steps[0].action.values)

    if n_jp != robot.n_joints:
        result.add_warning(
            f"joint_positions dim ({n_jp}) != robot.n_joints ({robot.n_joints})"
        )
    if n_act != robot.n_actuators:
        result.add_warning(
            f"action dim ({n_act}) != robot.n_actuators ({robot.n_actuators})"
        )


def _check_physical_plausibility(episode: Episode, result: ValidationResult) -> None:
    """Check for physically implausible values."""
    if not episode.steps:
        return

    robot = episode.metadata.robot
    try:
        np_data = episode.to_numpy()
    except (ValueError, TypeError) as e:
        result.add_error(f"Cannot convert to numpy (likely inconsistent dimensions): {e}")
        return

    # Joint position range check
    jp = np_data["joint_positions"]
    if robot.joints:
        for j_idx, jinfo in enumerate(robot.joints):
            if j_idx >= jp.shape[1]:
                break
            col = jp[:, j_idx]
            if np.any(col < jinfo.lower_limit - 0.1) or np.any(col > jinfo.upper_limit + 0.1):
                result.add_warning(
                    f"Joint '{jinfo.name}' exceeds limits "
                    f"[{jinfo.lower_limit:.2f}, {jinfo.upper_limit:.2f}], "
                    f"actual range [{col.min():.2f}, {col.max():.2f}]"
                )

    # Check for NaN / Inf
    for key, arr in np_data.items():
        if np.any(np.isnan(arr)):
            result.add_error(f"NaN detected in {key}")
        if np.any(np.isinf(arr)):
            result.add_error(f"Inf detected in {key}")

    # Check actions are within reasonable range
    actions = np_data["actions"]
    if np.any(np.abs(actions) > 1000):
        result.add_warning(f"Large action values detected (max={np.abs(actions).max():.1f})")

    # Energy sanity check
    jv = np_data["joint_velocities"]
    if actions.shape == jv.shape:
        power = np.abs(actions * jv)
        total_energy = np.sum(power) / episode.metadata.sampling_rate_hz
        if total_energy > 1e6:
            result.add_warning(f"Extremely high energy: {total_energy:.0f} J")


def _check_temporal_consistency(episode: Episode, result: ValidationResult) -> None:
    """Check timestamps are monotonically increasing."""
    timestamps = [s.timestamp for s in episode.steps]

    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            result.add_error(
                f"Non-monotonic timestamp at step {i}: "
                f"{timestamps[i]:.4f} <= {timestamps[i-1]:.4f}"
            )
            break

    # Check sampling rate consistency
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        expected_dt = 1.0 / episode.metadata.sampling_rate_hz
        dt_deviation = np.abs(dt - expected_dt)
        if np.any(dt_deviation > expected_dt * 0.5):
            result.add_warning(
                f"Irregular sampling: expected dt={expected_dt:.4f}s, "
                f"actual range [{dt.min():.4f}, {dt.max():.4f}]"
            )
