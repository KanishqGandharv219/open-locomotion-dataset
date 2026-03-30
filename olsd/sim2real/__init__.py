"""OLSD sim-to-real helpers and training-time utilities."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olsd.sim2real.alignment_eval import (
        AlignmentReport,
        evaluate_alignment,
        evaluate_episode_alignment,
        save_alignment_report,
    )
    from olsd.sim2real.domain_config import (
        DomainRandomConfig,
        RobotSim2RealConfig,
        derive_domain_randomization,
        load_sim2real_config,
        save_sim2real_config,
    )
    from olsd.sim2real.system_id import (
        SimParams,
        SimulatorBackend,
        TemplateReplayBackend,
        identify_params,
    )
    from olsd.sim2real.terrain import generate_terrain_xml

__all__ = [
    "AlignmentReport",
    "DomainRandomConfig",
    "RobotSim2RealConfig",
    "SimParams",
    "SimulatorBackend",
    "TemplateReplayBackend",
    "derive_domain_randomization",
    "evaluate_alignment",
    "evaluate_episode_alignment",
    "generate_terrain_xml",
    "identify_params",
    "load_sim2real_config",
    "save_alignment_report",
    "save_sim2real_config",
]

_EXPORTS = {
    "AlignmentReport": "olsd.sim2real.alignment_eval",
    "evaluate_alignment": "olsd.sim2real.alignment_eval",
    "evaluate_episode_alignment": "olsd.sim2real.alignment_eval",
    "save_alignment_report": "olsd.sim2real.alignment_eval",
    "DomainRandomConfig": "olsd.sim2real.domain_config",
    "RobotSim2RealConfig": "olsd.sim2real.domain_config",
    "derive_domain_randomization": "olsd.sim2real.domain_config",
    "load_sim2real_config": "olsd.sim2real.domain_config",
    "save_sim2real_config": "olsd.sim2real.domain_config",
    "SimParams": "olsd.sim2real.system_id",
    "SimulatorBackend": "olsd.sim2real.system_id",
    "TemplateReplayBackend": "olsd.sim2real.system_id",
    "identify_params": "olsd.sim2real.system_id",
    "generate_terrain_xml": "olsd.sim2real.terrain",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
