"""Helpers for strict Go1 baseline comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from olsd.schema import TerrainType
from olsd.sim2real.domain_config import load_sim2real_config
from olsd.sim2real.go1_env import Go1SimEnv
from olsd.sim2real.system_id import SimParams


DEFAULT_SELECTED_BASELINE = {
    "baseline_id": "walk_these_ways_pretrain_v0",
    "label": "Improbable-AI walk-these-ways pretrain-v0",
    "source_repo": "https://github.com/Improbable-AI/walk-these-ways",
    "checkpoint_subpath": "runs/gait-conditioned-agility/pretrain-v0/train/025417.456545",
    "status": "selected_not_integrated",
    "notes": [
        "Chosen as the first external Unitree Go1 baseline because it is public, Go1-specific, PPO-based, and ships a documented pretrained run.",
        "The strict head-to-head protocol must use shared physical metrics only; reward returns are not comparable across repositories.",
    ],
}

SHARED_GO1_METRICS = [
    "success_rate",
    "episode_length_mean",
    "episode_length_std",
    "forward_velocity_mean",
    "forward_velocity_std",
    "fall_count",
    "fall_rate",
]

WTW_JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

WTW_DEFAULT_COMMAND_PROFILE = {
    "x_vel_cmd": 0.6,
    "y_vel_cmd": 0.0,
    "yaw_vel_cmd": 0.0,
    "body_height_cmd": 0.0,
    "step_frequency_cmd": 3.0,
    "gait_phase_cmd": 0.5,
    "gait_offset_cmd": 0.0,
    "gait_bound_cmd": 0.0,
    "gait_duration_cmd": 0.5,
    "footswing_height_cmd": 0.08,
    "pitch_cmd": 0.0,
    "roll_cmd": 0.0,
    "stance_width_cmd": 0.25,
    "stance_length_cmd": 0.4,
    "aux_reward_cmd": 0.0,
}


def default_go1_terrain_params(terrain: TerrainType) -> dict[str, float]:
    """Return the canonical terrain parameters used for Phase 2 Go1 runs."""
    if terrain == TerrainType.SLOPE:
        return {"angle_deg": 5.0, "friction": 1.5}
    if terrain == TerrainType.STAIRS:
        return {"step_height": 0.04, "step_width": 0.4, "friction": 1.5}
    return {}


def resolve_go1_sim_params(sim2real_config: str | Path | None) -> SimParams:
    """Load Go1 sim params from YAML or fall back to defaults."""
    if sim2real_config is None:
        return SimParams.default()
    return load_sim2real_config(sim2real_config).identified_params


def evaluate_go1_sb3_policy(
    policy_path: str | Path,
    terrain: TerrainType,
    sim_params: SimParams,
    n_eval_episodes: int = 20,
    horizon: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate an SB3 PPO policy on the strict shared-metric protocol."""
    from stable_baselines3 import PPO

    model = PPO.load(str(policy_path), device="cpu")
    terrain_params = default_go1_terrain_params(terrain)
    episode_records: list[dict[str, float | int | bool]] = []

    for episode_idx in range(n_eval_episodes):
        env = Go1SimEnv(
            terrain_type=terrain,
            terrain_params=terrain_params,
            sim_params=sim_params,
            max_steps=horizon,
        )
        observation, _ = env.reset(seed=seed + episode_idx)
        step_count = 0
        terminated = False
        truncated = False
        velocity_sum = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            velocity_sum += float(info.get("x_velocity", 0.0))
            step_count += 1

        env.close()
        mean_velocity = velocity_sum / max(step_count, 1)
        episode_records.append(
            {
                "success": not bool(terminated),
                "fall": bool(terminated),
                "episode_length": step_count,
                "forward_velocity_mean": float(mean_velocity),
            }
        )

    return summarize_episode_records(episode_records)


def evaluate_walk_these_ways_policy(
    *,
    repo_root: str | Path,
    terrain: TerrainType,
    sim_params: SimParams,
    n_eval_episodes: int = 20,
    horizon: int = 1000,
    seed: int = 0,
    command_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate the public walk-these-ways pretrain-v0 checkpoint through an OLSD adapter."""
    adapter = WalkTheseWaysAdapter.from_repo_root(
        repo_root=repo_root,
        command_profile=command_profile,
    )
    terrain_params = default_go1_terrain_params(terrain)
    episode_records: list[dict[str, float | int | bool]] = []

    for episode_idx in range(n_eval_episodes):
        env = Go1SimEnv(
            terrain_type=terrain,
            terrain_params=terrain_params,
            sim_params=sim_params,
            max_steps=horizon,
        )
        env.reset(seed=seed + episode_idx)
        adapter.reset()
        adapter.prepare_env(env)
        step_count = 0
        terminated = False
        truncated = False
        velocity_sum = 0.0

        while not (terminated or truncated):
            action = adapter.predict()
            _, _, terminated, truncated, info = adapter.step(env, action)
            velocity_sum += float(info.get("x_velocity", 0.0))
            step_count += 1

        env.close()
        mean_velocity = velocity_sum / max(step_count, 1)
        episode_records.append(
            {
                "success": not bool(terminated),
                "fall": bool(terminated),
                "episode_length": step_count,
                "forward_velocity_mean": float(mean_velocity),
            }
        )

    summary = summarize_episode_records(episode_records)
    summary["adapter"] = "walk_these_ways_native_jit"
    summary["command_profile"] = adapter.command_profile
    summary["reset_pose"] = "checkpoint_default_dof_pos"
    return summary


def summarize_episode_records(
    episode_records: list[dict[str, float | int | bool]],
) -> dict[str, Any]:
    """Aggregate shared physical metrics from per-episode records."""
    if not episode_records:
        raise ValueError("episode_records must be non-empty")

    success_values = [1.0 if bool(record["success"]) else 0.0 for record in episode_records]
    fall_values = [1.0 if bool(record["fall"]) else 0.0 for record in episode_records]
    episode_lengths = [float(record["episode_length"]) for record in episode_records]
    forward_velocities = [float(record["forward_velocity_mean"]) for record in episode_records]

    return {
        "episode_count": len(episode_records),
        "shared_metrics": SHARED_GO1_METRICS,
        "success_rate": _mean(success_values),
        "episode_length_mean": _mean(episode_lengths),
        "episode_length_std": _std(episode_lengths),
        "forward_velocity_mean": _mean(forward_velocities),
        "forward_velocity_std": _std(forward_velocities),
        "fall_count": int(round(sum(fall_values))),
        "fall_rate": _mean(fall_values),
    }


def build_go1_head_to_head_report(
    *,
    baselines: dict[str, dict[str, Any]],
    selected_external_baseline: dict[str, Any] | None = None,
    n_eval_episodes: int = 20,
    horizon: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the canonical head-to-head report payload."""
    return {
        "schema_version": "1.0",
        "robot_id": "unitree_go1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "terrains": ["flat", "slope", "stairs"],
            "n_eval_episodes": n_eval_episodes,
            "horizon": horizon,
            "seed": seed,
            "shared_metrics": SHARED_GO1_METRICS,
            "reward_policy": "excluded_from_comparison",
        },
        "selected_external_baseline": selected_external_baseline or dict(DEFAULT_SELECTED_BASELINE),
        "baselines": baselines,
    }


def save_go1_head_to_head_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Persist the comparison report as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return output


class WalkTheseWaysAdapter:
    """Compatibility adapter for the public walk-these-ways Go1 pretrain-v0 checkpoint."""

    def __init__(
        self,
        *,
        body,
        adaptation_module,
        default_dof_pos: np.ndarray,
        commands_scale: np.ndarray,
        history_length: int,
        obs_dim: int,
        action_scale: float,
        hip_scale_reduction: float,
        command_profile: dict[str, float] | None = None,
    ):
        import torch

        self._torch = torch
        self.body = body.eval()
        self.adaptation_module = adaptation_module.eval()
        self.default_dof_pos = default_dof_pos.astype(np.float32)
        self.commands_scale = commands_scale.astype(np.float32)
        self.history_length = int(history_length)
        self.obs_dim = int(obs_dim)
        self.action_scale = float(action_scale)
        self.hip_scale_reduction = float(hip_scale_reduction)
        self.command_profile = dict(command_profile or WTW_DEFAULT_COMMAND_PROFILE)
        self.command_vector = _build_wtw_command_vector(self.command_profile)
        self.reset()

    @classmethod
    def from_repo_root(
        cls,
        *,
        repo_root: str | Path,
        command_profile: dict[str, float] | None = None,
    ) -> "WalkTheseWaysAdapter":
        import torch

        root = Path(repo_root)
        run_dir = root / "runs" / "gait-conditioned-agility" / "pretrain-v0" / "train" / "025417.456545"
        checkpoint_dir = run_dir / "checkpoints"
        body = torch.jit.load(str(checkpoint_dir / "body_latest.jit"), map_location="cpu")
        adaptation_module = torch.jit.load(
            str(checkpoint_dir / "adaptation_module_latest.jit"),
            map_location="cpu",
        )

        with open(run_dir / "parameters.pkl", "rb") as handle:
            cfg = pickle.load(handle)["Cfg"]

        history_length = int(cfg["env"]["num_observation_history"])
        obs_dim = int(cfg["env"]["num_observations"])
        default_joint_angles = cfg["init_state"]["default_joint_angles"]
        default_dof_pos = np.array(
            [default_joint_angles[name] for name in WTW_JOINT_ORDER],
            dtype=np.float32,
        )
        obs_scales = cfg["obs_scales"]
        num_commands = int(cfg["commands"]["num_commands"])
        commands_scale = np.array(
            [
                obs_scales["lin_vel"],
                obs_scales["lin_vel"],
                obs_scales["ang_vel"],
                obs_scales["body_height_cmd"],
                obs_scales["gait_freq_cmd"],
                obs_scales["gait_phase_cmd"],
                obs_scales["gait_phase_cmd"],
                obs_scales["gait_phase_cmd"],
                obs_scales["gait_phase_cmd"],
                obs_scales["footswing_height_cmd"],
                obs_scales["body_pitch_cmd"],
                obs_scales["body_roll_cmd"],
                obs_scales["stance_width_cmd"],
                obs_scales["stance_length_cmd"],
                obs_scales["aux_reward_cmd"],
            ],
            dtype=np.float32,
        )[:num_commands]

        return cls(
            body=body,
            adaptation_module=adaptation_module,
            default_dof_pos=default_dof_pos,
            commands_scale=commands_scale,
            history_length=history_length,
            obs_dim=obs_dim,
            action_scale=float(cfg["control"]["action_scale"]),
            hip_scale_reduction=float(cfg["control"]["hip_scale_reduction"]),
            command_profile=command_profile,
        )

    def reset(self) -> None:
        self.obs_history = self._torch.zeros(
            (1, self.obs_dim * self.history_length),
            dtype=self._torch.float32,
        )
        self.current_action = np.zeros(12, dtype=np.float32)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.gait_index = 0.0

    def prepare_env(self, env: Go1SimEnv) -> np.ndarray:
        """Align the local Go1 env with the checkpoint's nominal joint pose before rollout."""
        return env.settle_joint_positions(self.default_dof_pos)

    def predict(self) -> np.ndarray:
        with self._torch.no_grad():
            latent = self.adaptation_module.forward(self.obs_history)
            action = self.body.forward(self._torch.cat((self.obs_history, latent), dim=-1))
        return np.asarray(action.squeeze(0).cpu().numpy(), dtype=np.float32)

    def step(self, env: Go1SimEnv, action: np.ndarray):
        target_positions = self.action_to_target_positions(action)
        transition = env.step_target_positions(target_positions)
        self.previous_action = self.current_action.copy()
        self.current_action = np.asarray(action, dtype=np.float32).copy()
        self.gait_index = float(
            np.remainder(
                self.gait_index + env.control_dt * self.command_vector[4],
                1.0,
            )
        )
        scalar_obs = self.build_scalar_obs(env)
        obs_tensor = self._torch.as_tensor(scalar_obs, dtype=self._torch.float32).view(1, -1)
        self.obs_history = self._torch.cat(
            (self.obs_history[:, self.obs_dim :], obs_tensor),
            dim=-1,
        )
        return transition

    def action_to_target_positions(self, action: np.ndarray) -> np.ndarray:
        scaled = np.asarray(action[:12], dtype=np.float32) * self.action_scale
        scaled[[0, 3, 6, 9]] *= self.hip_scale_reduction
        return self.default_dof_pos + scaled

    def build_scalar_obs(self, env: Go1SimEnv) -> np.ndarray:
        projected_gravity = _rotate_world_vector_into_body_frame(
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            np.asarray(env.data.qpos[3:7], dtype=np.float32),
        )
        joint_pos_offset = np.asarray(env.data.qpos[7:19], dtype=np.float32) - self.default_dof_pos
        joint_vel = np.asarray(env.data.qvel[6:18], dtype=np.float32) * 0.05
        clock_inputs = self._clock_inputs()
        return np.concatenate(
            [
                projected_gravity,
                self.command_vector * self.commands_scale,
                joint_pos_offset,
                joint_vel,
                self.current_action,
                self.previous_action,
                clock_inputs,
            ],
            axis=0,
        ).astype(np.float32)

    def _clock_inputs(self) -> np.ndarray:
        phase = self.command_vector[5]
        offset = self.command_vector[6]
        bound = self.command_vector[7]
        foot_indices = np.array(
            [
                self.gait_index + phase + offset + bound,
                self.gait_index + offset,
                self.gait_index + bound,
                self.gait_index + phase,
            ],
            dtype=np.float32,
        )
        foot_indices = np.remainder(foot_indices, 1.0)
        return np.sin(2.0 * np.pi * foot_indices).astype(np.float32)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(variance ** 0.5)


def _build_wtw_command_vector(command_profile: dict[str, float]) -> np.ndarray:
    """Create the 15-command walk-these-ways command vector."""
    return np.array(
        [
            command_profile["x_vel_cmd"],
            command_profile["y_vel_cmd"],
            command_profile["yaw_vel_cmd"],
            command_profile["body_height_cmd"],
            command_profile["step_frequency_cmd"],
            command_profile["gait_phase_cmd"],
            command_profile["gait_offset_cmd"],
            command_profile["gait_bound_cmd"],
            command_profile["gait_duration_cmd"],
            command_profile["footswing_height_cmd"],
            command_profile["pitch_cmd"],
            command_profile["roll_cmd"],
            command_profile["stance_width_cmd"],
            command_profile["stance_length_cmd"],
            command_profile["aux_reward_cmd"],
        ],
        dtype=np.float32,
    )


def _rotate_world_vector_into_body_frame(vector: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Equivalent to quat_rotate_inverse for MuJoCo-style [w, x, y, z] quaternions."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float32)
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation.T @ np.asarray(vector, dtype=np.float32)
