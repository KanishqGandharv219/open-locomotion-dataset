"""Sim-to-real identified parameter and domain randomization config helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from olsd.sim2real.system_id import SimParams


@dataclass
class DomainRandomConfig:
    """Domain randomization ranges centered on identified params."""

    friction_range: tuple[float, float]
    mass_scale_range: tuple[float, float]
    joint_damping_range: tuple[float, float]
    joint_armature_range: tuple[float, float]
    kp_range: tuple[float, float]
    kd_range: tuple[float, float]
    latency_range_ms: tuple[float, float]
    observation_noise_std: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DomainRandomConfig":
        return cls(
            friction_range=tuple(data["friction_range"]),
            mass_scale_range=tuple(data["mass_scale_range"]),
            joint_damping_range=tuple(data["joint_damping_range"]),
            joint_armature_range=tuple(data["joint_armature_range"]),
            kp_range=tuple(data["kp_range"]),
            kd_range=tuple(data["kd_range"]),
            latency_range_ms=tuple(data["latency_range_ms"]),
            observation_noise_std=float(data["observation_noise_std"]),
        )


@dataclass
class RobotSim2RealConfig:
    """YAML-ready sim-to-real config persisted under configs/sim2real/."""

    robot_id: str
    identified_params: SimParams
    domain_randomization: DomainRandomConfig
    template_only: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "identified_params": self.identified_params.to_dict(),
            "domain_randomization": self.domain_randomization.to_dict(),
            "template_only": self.template_only,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RobotSim2RealConfig":
        return cls(
            robot_id=str(data["robot_id"]),
            identified_params=SimParams.from_dict(data["identified_params"]),
            domain_randomization=DomainRandomConfig.from_dict(data["domain_randomization"]),
            template_only=bool(data.get("template_only", False)),
            notes=list(data.get("notes", [])),
        )


def derive_domain_randomization(
    params: SimParams,
    relative_margin: float = 0.15,
    latency_margin_ms: float = 3.0,
    min_noise_std: float = 0.005,
) -> DomainRandomConfig:
    """Derive DR ranges around an identified parameter set."""
    return DomainRandomConfig(
        friction_range=_bounded_range(params.global_friction, relative_margin, 0.1, 4.0),
        mass_scale_range=_bounded_range(params.mass_scale, relative_margin, 0.5, 1.5),
        joint_damping_range=_bounded_range(params.joint_damping_scale, relative_margin, 0.1, 3.0),
        joint_armature_range=_bounded_range(params.joint_armature_scale, relative_margin, 0.1, 3.0),
        kp_range=_bounded_range(params.kp_scale, relative_margin, 0.1, 3.0),
        kd_range=_bounded_range(params.kd_scale, relative_margin, 0.1, 3.0),
        latency_range_ms=_bounded_range(
            params.actuator_latency_ms,
            0.0,
            0.0,
            50.0,
            latency_margin_ms,
        ),
        observation_noise_std=max(
            min_noise_std,
            params.observation_noise_std * (1.0 + relative_margin),
        ),
    )


def save_sim2real_config(config: RobotSim2RealConfig, output_path: str | Path) -> Path:
    """Persist a RobotSim2RealConfig to YAML."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    return output


def load_sim2real_config(path: str | Path) -> RobotSim2RealConfig:
    """Load a RobotSim2RealConfig from YAML."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return RobotSim2RealConfig.from_dict(raw)


def _bounded_range(
    center: float,
    relative_margin: float,
    lower: float,
    upper: float,
    absolute_margin: float | None = None,
) -> tuple[float, float]:
    """Build a clipped interval around a scalar."""
    margin = absolute_margin if absolute_margin is not None else abs(center) * relative_margin
    return (
        max(lower, center - margin),
        min(upper, center + margin),
    )
