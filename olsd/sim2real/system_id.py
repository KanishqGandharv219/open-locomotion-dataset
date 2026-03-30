"""Reduced-parameter system identification for OLSD sim-to-real alignment."""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from olsd.pipeline.ingest import load_robot_by_id
from olsd.schema import DataSource, Episode, RobotSpec
from olsd.sim2real._io import clone_episode_with_arrays, load_episodes_from_path, load_mjcf_xml
from olsd.sim2real.alignment_eval import evaluate_alignment, save_alignment_report

logger = logging.getLogger(__name__)


@dataclass
class SimParams:
    """Reduced system-identification parameter set."""

    global_friction: float
    mass_scale: float
    joint_damping_scale: float
    joint_armature_scale: float
    kp_scale: float
    kd_scale: float
    actuator_latency_ms: float
    observation_noise_std: float

    def __post_init__(self) -> None:
        for field_name, value in self.to_dict().items():
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
        if self.global_friction <= 0 or self.mass_scale <= 0:
            raise ValueError("global_friction and mass_scale must be positive")
        if self.actuator_latency_ms < 0 or self.observation_noise_std < 0:
            raise ValueError("latency and observation_noise_std must be non-negative")

    @classmethod
    def default(cls) -> "SimParams":
        return cls(
            global_friction=1.0,
            mass_scale=1.0,
            joint_damping_scale=1.0,
            joint_armature_scale=1.0,
            kp_scale=1.0,
            kd_scale=1.0,
            actuator_latency_ms=2.0,
            observation_noise_std=0.01,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SimParams":
        return cls(**{key: float(value) for key, value in data.items()})

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "SimParams":
        values = np.asarray(vector, dtype=np.float64)
        return cls(
            global_friction=float(values[0]),
            mass_scale=float(values[1]),
            joint_damping_scale=float(values[2]),
            joint_armature_scale=float(values[3]),
            kp_scale=float(values[4]),
            kd_scale=float(values[5]),
            actuator_latency_ms=float(values[6]),
            observation_noise_std=float(values[7]),
        )

    @classmethod
    def bounds(cls) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0], dtype=np.float64)
        upper = np.array([4.0, 1.5, 3.0, 3.0, 3.0, 3.0, 50.0, 0.2], dtype=np.float64)
        return lower, upper

    def to_dict(self) -> dict[str, float]:
        return {
            "global_friction": self.global_friction,
            "mass_scale": self.mass_scale,
            "joint_damping_scale": self.joint_damping_scale,
            "joint_armature_scale": self.joint_armature_scale,
            "kp_scale": self.kp_scale,
            "kd_scale": self.kd_scale,
            "actuator_latency_ms": self.actuator_latency_ms,
            "observation_noise_std": self.observation_noise_std,
        }

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.global_friction,
                self.mass_scale,
                self.joint_damping_scale,
                self.joint_armature_scale,
                self.kp_scale,
                self.kd_scale,
                self.actuator_latency_ms,
                self.observation_noise_std,
            ],
            dtype=np.float64,
        )


class SimulatorBackend(Protocol):
    """Pluggable trajectory replay backend for system identification."""

    def rollout_episode(
        self,
        reference_episode: Episode,
        params: SimParams,
        robot_config: RobotSpec,
        mjcf_xml: str | None,
    ) -> Episode:
        ...


class TemplateReplayBackend:
    """Fallback backend that perturbs reference trajectories using the candidate params."""

    def rollout_episode(
        self,
        reference_episode: Episode,
        params: SimParams,
        robot_config: RobotSpec,
        mjcf_xml: str | None,
    ) -> Episode:
        data = reference_episode.to_numpy()
        joint_positions = data["joint_positions"].astype(np.float64)
        joint_velocities = data["joint_velocities"].astype(np.float64)
        actions = data["actions"].astype(np.float64)
        dt = 1.0 / max(reference_episode.metadata.sampling_rate_hz, 1e-6)

        delayed_actions = _apply_latency(actions, params.actuator_latency_ms, dt)
        friction_term = np.tanh(delayed_actions) * (params.global_friction - 1.0) * 0.015
        damping_term = joint_velocities * (2.0 - params.joint_damping_scale)
        armature_term = np.gradient(joint_velocities, axis=0) * params.joint_armature_scale * 0.01
        gain_term = np.tanh(delayed_actions * params.kp_scale) * 0.05
        derivative_term = np.gradient(delayed_actions, axis=0) * params.kd_scale * 0.005

        predicted_velocities = (damping_term + gain_term - derivative_term) / max(params.mass_scale, 1e-6)
        predicted_positions = joint_positions + (predicted_velocities * dt) + friction_term - armature_term

        if params.observation_noise_std > 0:
            noise = np.random.default_rng(0).normal(
                0.0,
                params.observation_noise_std,
                size=predicted_positions.shape,
            )
            predicted_positions = predicted_positions + noise
            predicted_velocities = predicted_velocities + noise

        simulated = clone_episode_with_arrays(
            reference_episode,
            predicted_positions.astype(np.float32),
            predicted_velocities.astype(np.float32),
            delayed_actions.astype(np.float32),
        )
        simulated.metadata = deepcopy(reference_episode.metadata)
        simulated.metadata.source = DataSource.SIMULATION
        simulated.metadata.dynamics_params = params.to_dict()
        simulated.metadata.simulator = simulated.metadata.simulator or "template_replay"
        return simulated


class Go1ReplayBackend:
    """Replay Go1 action sequences through the reusable MuJoCo env."""

    def rollout_episode(
        self,
        reference_episode: Episode,
        params: SimParams,
        robot_config: RobotSpec,
        mjcf_xml: str | None,
    ) -> Episode:
        from olsd.sim2real.go1_env import replay_go1_episode

        action_sequence = np.array([step.action.values for step in reference_episode.steps], dtype=np.float32)
        terrain = reference_episode.metadata.terrain.terrain_type
        return replay_go1_episode(
            action_sequence=action_sequence,
            sim_params=params,
            terrain_type=terrain,
            terrain_params={},
            robot_spec=robot_config,
        )


def identify_params(
    real_trajectories: list[Episode],
    robot_config: RobotSpec,
    mjcf_xml: str | None,
    n_generations: int = 50,
    population_size: int = 30,
    objective_fn=None,
    simulator_backend: SimulatorBackend | None = None,
    initial_params: SimParams | None = None,
    seed: int = 0,
) -> SimParams:
    """Run CMA-ES (or a small fallback search) to fit SimParams."""
    if not real_trajectories:
        raise ValueError("real_trajectories must be non-empty")

    init = initial_params or SimParams.default()
    backend = simulator_backend or _default_backend(robot_config.robot_id, mjcf_xml)
    lower_bounds, upper_bounds = SimParams.bounds()
    best_params = init
    best_score = float("inf")

    def score_vector(vector: np.ndarray) -> float:
        nonlocal best_params, best_score
        clipped = np.clip(vector, lower_bounds, upper_bounds)
        params = SimParams.from_vector(clipped)
        score = float(_score_candidate(
            params=params,
            real_trajectories=real_trajectories,
            robot_config=robot_config,
            mjcf_xml=mjcf_xml,
            backend=backend,
            objective_fn=objective_fn,
        ))
        if score < best_score:
            best_score = score
            best_params = params
        return score

    try:
        import cma

        sigma0 = 0.15
        es = cma.CMAEvolutionStrategy(
            init.to_vector().tolist(),
            sigma0,
            {
                "bounds": [lower_bounds.tolist(), upper_bounds.tolist()],
                "popsize": population_size,
                "seed": seed,
                "verbose": -9,
            },
        )
        for generation_idx in range(n_generations):
            solutions = np.array(es.ask(), dtype=np.float64)
            scores = [score_vector(solution) for solution in solutions]
            es.tell(solutions.tolist(), scores)
            logger.info(
                "system_id generation %d/%d best_score=%.5f",
                generation_idx + 1,
                n_generations,
                best_score,
            )
    except ImportError:
        rng = np.random.default_rng(seed)
        current_center = init.to_vector()
        for generation_idx in range(n_generations):
            candidates = []
            for _ in range(population_size):
                noise = rng.normal(0.0, 0.12, size=current_center.shape)
                candidates.append(np.clip(current_center + noise, lower_bounds, upper_bounds))
            scores = [score_vector(candidate) for candidate in candidates]
            current_center = candidates[int(np.argmin(scores))]
            logger.info(
                "fallback system_id generation %d/%d best_score=%.5f",
                generation_idx + 1,
                n_generations,
                best_score,
            )

    return best_params


def simulate_episodes(
    real_trajectories: list[Episode],
    params: SimParams,
    robot_config: RobotSpec,
    mjcf_xml: str | None,
    simulator_backend: SimulatorBackend | None = None,
) -> list[Episode]:
    """Replay a list of episodes under the provided sim parameters."""
    backend = simulator_backend or _default_backend(robot_config.robot_id, mjcf_xml)
    return [
        backend.rollout_episode(reference_episode, params, robot_config, mjcf_xml)
        for reference_episode in real_trajectories
    ]


def _score_candidate(
    params: SimParams,
    real_trajectories: list[Episode],
    robot_config: RobotSpec,
    mjcf_xml: str | None,
    backend: SimulatorBackend,
    objective_fn=None,
) -> float:
    """Score a candidate parameter vector."""
    if objective_fn is not None:
        return float(objective_fn(params))

    sim_trajectories = simulate_episodes(
        real_trajectories=real_trajectories,
        params=params,
        robot_config=robot_config,
        mjcf_xml=mjcf_xml,
        simulator_backend=backend,
    )
    report = evaluate_alignment(real_trajectories, sim_trajectories)
    normalized_dtw = report["trajectory_dtw"] / max(report["mean_shared_steps"], 1.0)
    velocity_penalty = 1.0 - report["velocity_correlation"]
    return float(report["joint_rmse"] + velocity_penalty + normalized_dtw)


def _default_backend(robot_id: str, mjcf_xml: str | None) -> SimulatorBackend:
    """Pick the most capable backend available for a robot."""
    robot_key = robot_id.lower()
    if "go1" in robot_key:
        return Go1ReplayBackend()
    return TemplateReplayBackend()


def _apply_latency(actions: np.ndarray, latency_ms: float, dt: float) -> np.ndarray:
    """Apply actuator delay by shifting the action sequence."""
    delayed = np.array(actions, copy=True)
    if latency_ms <= 0 or dt <= 0:
        return delayed

    delay_steps = int(round((latency_ms / 1000.0) / dt))
    if delay_steps <= 0:
        return delayed
    delayed[delay_steps:] = actions[:-delay_steps]
    delayed[:delay_steps] = actions[0]
    return delayed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reduced-parameter system identification")
    parser.add_argument("--robot", required=True, help="Robot id, e.g. go1 or anymal_d")
    parser.add_argument("--real-data", required=True, help="Path to real/reference episodes")
    parser.add_argument("--mjcf-path", default=None, help="Optional MJCF path or raw XML")
    parser.add_argument("--output", required=True, help="Output YAML path")
    parser.add_argument("--generations", type=int, default=50, help="CMA-ES generations")
    parser.add_argument("--population", type=int, default=30, help="CMA-ES population size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from olsd.sim2real.domain_config import (
        RobotSim2RealConfig,
        derive_domain_randomization,
        save_sim2real_config,
    )

    robot_config = load_robot_by_id(args.robot)
    mjcf_xml = load_mjcf_xml(args.mjcf_path)
    real_trajectories = load_episodes_from_path(args.real_data, robot_id=args.robot)
    backend = _default_backend(robot_config.robot_id, mjcf_xml)
    params = identify_params(
        real_trajectories=real_trajectories,
        robot_config=robot_config,
        mjcf_xml=mjcf_xml,
        n_generations=args.generations,
        population_size=args.population,
        simulator_backend=backend,
        seed=args.seed,
    )
    sim_trajectories = simulate_episodes(
        real_trajectories=real_trajectories,
        params=params,
        robot_config=robot_config,
        mjcf_xml=mjcf_xml,
        simulator_backend=backend,
    )
    report = evaluate_alignment(real_trajectories, sim_trajectories)

    template_only = "go1" in robot_config.robot_id.lower()
    notes = []
    if template_only:
        notes.append("Template-only config derived from simulated Go1 reference data.")
    elif real_trajectories and all(ep.metadata.source == DataSource.HARDWARE for ep in real_trajectories):
        notes.append("Real-data alignment derived from local GrandTour ANYmal episodes.")
    else:
        notes.append("Sim-to-real config generated from local reference episodes.")
    if isinstance(backend, TemplateReplayBackend) and "go1" not in robot_config.robot_id.lower():
        notes.append(
            "This run used the reduced kinematic replay backend on real ANYmal episodes; "
            "the staged ANYmal asset is available locally for future physics-backed replay."
        )
    if mjcf_xml is None:
        notes.append("No MJCF was supplied; backend replay fell back to a template dynamics model.")

    config = RobotSim2RealConfig(
        robot_id=robot_config.robot_id,
        identified_params=params,
        domain_randomization=derive_domain_randomization(params),
        template_only=template_only,
        notes=notes,
    )
    output_path = save_sim2real_config(config, args.output)
    report_path = Path(args.output).with_suffix(".alignment.json")
    save_alignment_report(report, report_path)

    logger.info(
        "Saved sim2real config to %s (joint_rmse=%.4f, velocity_corr=%.4f)",
        output_path,
        report["joint_rmse"],
        report["velocity_correlation"],
    )


if __name__ == "__main__":
    main()
