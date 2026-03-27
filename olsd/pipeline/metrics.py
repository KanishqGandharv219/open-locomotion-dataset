"""
OLSD Gait Metrics — Auto-compute locomotion quality metrics from trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from olsd.schema import Episode


@dataclass
class GaitMetrics:
    """Computed gait metrics for a single episode."""

    # Speed
    mean_forward_speed: float = 0.0  # m/s
    speed_variance: float = 0.0

    # Gait temporal
    stride_frequency: float = 0.0  # Hz
    cadence: float = 0.0  # steps/min
    duty_cycle: float | None = None  # fraction of stride in stance

    # Energy
    total_energy_joules: float = 0.0
    energy_per_meter: float = 0.0  # J/m (cost of transport proxy)
    mean_power_watts: float = 0.0

    # Stability
    joint_position_variance: float = 0.0
    joint_velocity_variance: float = 0.0
    smoothness_index: float = 0.0  # lower = smoother

    # Actions
    action_magnitude_mean: float = 0.0
    action_rate_of_change: float = 0.0  # jerk-like measure

    # Duration
    duration_seconds: float = 0.0
    n_steps: int = 0

    def to_dict(self) -> dict[str, float | int | None]:
        """Convert to a flat dictionary."""
        return {
            "mean_forward_speed": self.mean_forward_speed,
            "speed_variance": self.speed_variance,
            "stride_frequency": self.stride_frequency,
            "cadence": self.cadence,
            "duty_cycle": self.duty_cycle,
            "total_energy_joules": self.total_energy_joules,
            "energy_per_meter": self.energy_per_meter,
            "mean_power_watts": self.mean_power_watts,
            "joint_position_variance": self.joint_position_variance,
            "joint_velocity_variance": self.joint_velocity_variance,
            "smoothness_index": self.smoothness_index,
            "action_magnitude_mean": self.action_magnitude_mean,
            "action_rate_of_change": self.action_rate_of_change,
            "duration_seconds": self.duration_seconds,
            "n_steps": self.n_steps,
        }


def compute_metrics(episode: Episode) -> GaitMetrics:
    """
    Compute gait metrics from an OLSD Episode.

    Uses joint positions/velocities to estimate gait patterns,
    and actions/torques for energy computation.
    """
    data = episode.to_numpy()
    jp = data["joint_positions"]  # (T, n_joints)
    jv = data["joint_velocities"]  # (T, n_joints)
    actions = data["actions"]  # (T, n_act)
    timestamps = data["timestamps"]  # (T,)

    n_steps = len(timestamps)
    dt = 1.0 / episode.metadata.sampling_rate_hz
    duration = episode.duration_seconds

    metrics = GaitMetrics(n_steps=n_steps, duration_seconds=duration)

    if n_steps < 2:
        return metrics

    # ----- Speed (from base velocity if available, else from joint data) -----
    base_vels = _extract_base_velocities(episode)
    if base_vels is not None:
        metrics.mean_forward_speed = float(np.mean(np.abs(base_vels[:, 0])))
        metrics.speed_variance = float(np.var(base_vels[:, 0]))
    elif episode.metadata.actual_speed_mps is not None:
        metrics.mean_forward_speed = episode.metadata.actual_speed_mps

    # ----- Stride frequency (from dominant frequency of joint oscillations) -----
    if n_steps > 10:
        metrics.stride_frequency = _estimate_stride_frequency(jp, dt)
        if metrics.stride_frequency > 0:
            metrics.cadence = metrics.stride_frequency * 60.0  # strides/min

    # ----- Energy -----
    # Power ≈ |torque × velocity| (use actions as torque proxy if no torques)
    if actions.shape[1] == jv.shape[1]:
        power = np.abs(actions * jv)
    else:
        power = np.abs(actions) * np.mean(np.abs(jv), axis=1, keepdims=True)

    instant_power = np.sum(power, axis=1)
    metrics.total_energy_joules = float(np.sum(instant_power) * dt)
    metrics.mean_power_watts = float(np.mean(instant_power))

    distance = metrics.mean_forward_speed * duration
    if distance > 0.01:
        metrics.energy_per_meter = metrics.total_energy_joules / distance

    # ----- Stability -----
    metrics.joint_position_variance = float(np.mean(np.var(jp, axis=0)))
    metrics.joint_velocity_variance = float(np.mean(np.var(jv, axis=0)))

    # Smoothness: mean ||a(t+1) - a(t)||
    if n_steps > 1:
        action_diff = np.diff(actions, axis=0)
        metrics.action_rate_of_change = float(np.mean(np.linalg.norm(action_diff, axis=1)))
        # Spectral arc length smoothness (simplified: lower = smoother)
        metrics.smoothness_index = float(np.mean(np.abs(action_diff)))

    # ----- Action stats -----
    metrics.action_magnitude_mean = float(np.mean(np.linalg.norm(actions, axis=1)))

    # ----- Duty cycle (from contact data if available) -----
    metrics.duty_cycle = _estimate_duty_cycle(episode)

    return metrics


def compute_dataset_metrics(episodes: list[Episode]) -> dict[str, float]:
    """Compute aggregate metrics across a list of episodes."""
    all_metrics = [compute_metrics(ep) for ep in episodes]

    if not all_metrics:
        return {}

    keys = [
        "mean_forward_speed", "total_energy_joules", "energy_per_meter",
        "mean_power_watts", "smoothness_index", "stride_frequency",
    ]

    summary = {}
    for key in keys:
        values = [getattr(m, key) for m in all_metrics if getattr(m, key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))

    summary["total_episodes"] = len(episodes)
    summary["total_steps"] = sum(m.n_steps for m in all_metrics)
    summary["total_duration_seconds"] = sum(m.duration_seconds for m in all_metrics)

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_base_velocities(episode: Episode) -> np.ndarray | None:
    """Extract base velocity from episode if available."""
    vels = []
    for step in episode.steps:
        if step.observation.base_velocity is not None:
            vels.append(step.observation.base_velocity)
        else:
            return None
    return np.array(vels)


def _estimate_stride_frequency(joint_positions: np.ndarray, dt: float) -> float:
    """
    Estimate stride frequency from joint angle oscillations.
    Uses FFT on the first joint (typically a hip joint).
    """
    if joint_positions.shape[0] < 20:
        return 0.0

    # Use first joint (hip) as proxy for gait cycle
    sig = joint_positions[:, 0]
    sig = sig - np.mean(sig)  # detrend

    # FFT
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_mag = np.abs(np.fft.rfft(sig))

    # Ignore DC and very high frequencies
    valid = (freqs > 0.2) & (freqs < 15.0)
    if not np.any(valid):
        return 0.0

    fft_mag_valid = fft_mag[valid]
    freqs_valid = freqs[valid]

    dominant_idx = np.argmax(fft_mag_valid)
    dominant_freq = freqs_valid[dominant_idx]

    return float(dominant_freq)


def _estimate_duty_cycle(episode: Episode) -> float | None:
    """
    Estimate duty cycle from contact binary flags if available.
    Duty cycle = fraction of time in stance (contact = True).
    """
    contacts = []
    for step in episode.steps:
        if step.observation.contact_binary is not None:
            # Average across all feet
            contacts.append(np.mean(step.observation.contact_binary))
        else:
            return None

    return float(np.mean(contacts)) if contacts else None
