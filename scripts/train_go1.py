"""Train a Go1 locomotion policy with the reusable OLSD sim-to-real env."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from olsd.schema import TerrainType
from olsd.sim2real.domain_config import load_sim2real_config
from olsd.sim2real.go1_env import Go1SimEnv
from olsd.sim2real.system_id import SimParams

logger = logging.getLogger("olsd.train_go1")

POLICY_ARCH = [128, 128]
DEFAULT_TIMESTEPS = 2_000_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-4
DEFAULT_N_ENVS = 8
DEFAULT_PPO_N_STEPS = 1024
DEFAULT_GAMMA = 0.995
DEFAULT_ENT_COEF = 0.01
DEFAULT_EVAL_FREQ = 20_000
DEFAULT_CHECKPOINT_FREQ = 50_000


def _terrain_params_for(terrain: TerrainType) -> dict:
    if terrain == TerrainType.SLOPE:
        return {"angle_deg": 5.0, "friction": 1.5}
    if terrain == TerrainType.STAIRS:
        return {"step_height": 0.04, "step_width": 0.4, "friction": 1.5}
    return {}


def train_go1(
    terrain: TerrainType,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    ppo_n_steps: int = DEFAULT_PPO_N_STEPS,
    gamma: float = DEFAULT_GAMMA,
    ent_coef: float = DEFAULT_ENT_COEF,
    output_dir: str | Path = "checkpoints/go1",
    sim2real_config: str | Path | None = None,
    init_policy: str | Path | None = None,
    target_velocity: float = 0.5,
    seed: int = 42,
) -> dict:
    """Train PPO on the Go1 env and save a compact terrain summary."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    sim_params = _resolve_sim_params(sim2real_config)
    terrain_params = _terrain_params_for(terrain)
    out_dir = Path(output_dir) / terrain.value
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_env(env_seed: int):
        def _factory():
            env = Go1SimEnv(
                terrain_type=terrain,
                terrain_params=terrain_params,
                sim_params=sim_params,
                max_steps=1000,
                target_velocity=target_velocity,
                randomize_velocity=True,
                velocity_range=(0.3, 0.8),
            )
            env.reset(seed=env_seed)
            return env

        return _factory

    vec_env = DummyVecEnv([make_env(seed + idx) for idx in range(n_envs)])
    eval_env = DummyVecEnv([make_env(seed + 10_000)])
    best_model_dir = out_dir / "best_model"
    checkpoint_dir = out_dir / "checkpoints"
    callback = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(DEFAULT_CHECKPOINT_FREQ // max(n_envs, 1), 1),
                save_path=str(checkpoint_dir),
                name_prefix="go1_ppo",
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(out_dir / "eval_logs"),
                eval_freq=max(DEFAULT_EVAL_FREQ // max(n_envs, 1), 1),
                deterministic=True,
                render=False,
                n_eval_episodes=3,
            ),
        ]
    )
    if init_policy is not None:
        model = PPO.load(str(init_policy), env=vec_env, device="cpu")
        model.learning_rate = learning_rate
        model.batch_size = batch_size
        model.n_steps = ppo_n_steps
        model.gamma = gamma
        model.ent_coef = ent_coef
        model.seed = seed
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=ppo_n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": POLICY_ARCH},
            verbose=1,
            seed=seed,
            device="cpu",
        )
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)

    model_path = out_dir / "go1_ppo.zip"
    model.save(str(model_path.with_suffix("")))

    random_summary = evaluate_go1_policy(
        terrain=terrain,
        terrain_params=terrain_params,
        sim_params=sim_params,
        target_velocity=target_velocity,
        n_eval_episodes=5,
        seed=seed,
    )
    final_summary = evaluate_go1_policy(
        terrain=terrain,
        terrain_params=terrain_params,
        sim_params=sim_params,
        target_velocity=target_velocity,
        model=model,
        n_eval_episodes=5,
        seed=seed + 1000,
    )
    best_summary = None
    init_summary = None
    trained_summary = final_summary
    trained_source = "final_checkpoint"
    best_model_path = best_model_dir / "best_model.zip"
    if best_model_path.exists():
        best_model = PPO.load(str(best_model_path), device="cpu")
        best_summary = evaluate_go1_policy(
            terrain=terrain,
            terrain_params=terrain_params,
            sim_params=sim_params,
            target_velocity=target_velocity,
            model=best_model,
            n_eval_episodes=5,
            seed=seed + 2000,
        )
        if _policy_rank(best_summary) > _policy_rank(final_summary):
            trained_summary = best_summary
            trained_source = "best_checkpoint"
    if init_policy is not None:
        init_model = PPO.load(str(init_policy), device="cpu")
        init_summary = evaluate_go1_policy(
            terrain=terrain,
            terrain_params=terrain_params,
            sim_params=sim_params,
            target_velocity=target_velocity,
            model=init_model,
            n_eval_episodes=5,
            seed=seed + 3000,
        )
        if _policy_rank(init_summary) > _policy_rank(trained_summary):
            trained_summary = init_summary
            trained_source = "init_policy"

    selected_model_path = out_dir / "selected_model.zip"
    if trained_source == "best_checkpoint" and best_model_path.exists():
        shutil.copy2(best_model_path, selected_model_path)
    elif trained_source == "init_policy" and init_policy is not None:
        shutil.copy2(Path(init_policy), selected_model_path)
    else:
        shutil.copy2(model_path, selected_model_path)

    summary = build_training_summary(
        terrain=terrain,
        random_summary=random_summary,
        trained_summary=trained_summary,
        trained_source=trained_source,
        final_summary=final_summary,
        best_summary=best_summary,
        init_summary=init_summary,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ppo_n_steps=ppo_n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        sim2real_config=sim2real_config,
        init_policy=init_policy,
    )
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    eval_env.close()
    vec_env.close()
    logger.info("Saved Go1 checkpoint to %s", model_path)
    logger.info("Saved evaluation summary to %s", summary_path)
    return summary


def evaluate_go1_policy(
    terrain: TerrainType,
    terrain_params: dict,
    sim_params: SimParams,
    model=None,
    target_velocity: float = 0.5,
    n_eval_episodes: int = 5,
    seed: int = 0,
) -> dict:
    """Evaluate either a random policy or a trained PPO policy."""
    returns = []
    lengths = []
    successes = []

    for episode_idx in range(n_eval_episodes):
        env = Go1SimEnv(
            terrain_type=terrain,
            terrain_params=terrain_params,
            sim_params=sim_params,
            max_steps=1000,
            target_velocity=target_velocity,
            randomize_velocity=False,  # Fixed velocity for deterministic eval
        )
        observation, _ = env.reset(seed=seed + episode_idx)
        episode_return = 0.0
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            step_count += 1

        returns.append(float(episode_return))
        lengths.append(step_count)
        successes.append(0.0 if terminated else 1.0)
        env.close()

    return {
        "return_mean": float(sum(returns) / len(returns)),
        "return_std": float(_std(returns)),
        "episode_length_mean": float(sum(lengths) / len(lengths)),
        "success_rate": float(sum(successes) / len(successes)),
    }


def build_training_summary(
    terrain: TerrainType,
    random_summary: dict,
    trained_summary: dict,
    trained_source: str,
    final_summary: dict,
    best_summary: dict | None,
    init_summary: dict | None,
    total_timesteps: int,
    n_envs: int,
    batch_size: int,
    learning_rate: float,
    ppo_n_steps: int,
    gamma: float,
    ent_coef: float,
    sim2real_config: str | Path | None,
    init_policy: str | Path | None,
) -> dict:
    """Build the compact JSON summary requested for Phase 2."""
    return_gain = trained_summary["return_mean"] - random_summary["return_mean"]
    length_gain = (
        trained_summary["episode_length_mean"] - random_summary["episode_length_mean"]
    )
    success_gain = trained_summary["success_rate"] - random_summary["success_rate"]

    return {
        "robot_id": "unitree_go1",
        "terrain": terrain.value,
        "training": {
            "algorithm": "ppo",
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "policy_arch": POLICY_ARCH,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "ppo_n_steps": ppo_n_steps,
            "gamma": gamma,
            "ent_coef": ent_coef,
            "sim2real_config": str(sim2real_config) if sim2real_config else None,
            "init_policy": str(init_policy) if init_policy else None,
            "device": "cpu",
        },
        "evaluation": {
            "random": random_summary,
            "trained": trained_summary,
            "trained_source": trained_source,
            "final_checkpoint": final_summary,
            "best_checkpoint": best_summary,
            "init_policy_checkpoint": init_summary,
            "comparison": {
                "return_gain_vs_random": return_gain,
                "episode_length_gain_vs_random": length_gain,
                "success_rate_gain_vs_random": success_gain,
            },
            "normalization_note": "No fixed expert reference is defined yet for Go1, so this summary reports raw gains vs random instead of a normalized score and prefers the best checkpoint when it outperforms the final policy.",
        },
    }


def _resolve_sim_params(sim2real_config: str | Path | None) -> SimParams:
    """Load sim params from YAML or fall back to defaults."""
    if sim2real_config is None:
        return SimParams.default()
    return load_sim2real_config(sim2real_config).identified_params


def _std(values: list[float]) -> float:
    """Small helper to avoid pulling numpy into the script entrypoint."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _policy_rank(summary: dict) -> tuple[float, float, float]:
    """Rank policies by survivability first, then return."""
    return (
        float(summary["success_rate"]),
        float(summary["episode_length_mean"]),
        float(summary["return_mean"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Go1 PPO with the OLSD sim2real env")
    parser.add_argument("--terrain", default="flat", choices=["flat", "slope", "stairs"])
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--output-dir", default="checkpoints/go1")
    parser.add_argument("--sim2real-config", default=None)
    parser.add_argument("--init-policy", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--target-velocity", type=float, default=0.5,
                        help="Target forward velocity in m/s (default: 0.5). "
                             "During training, velocity is randomized per episode from [0.3, 0.8].")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    terrain = TerrainType(args.terrain)
    train_go1(
        terrain=terrain,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        sim2real_config=args.sim2real_config,
        init_policy=args.init_policy,
        target_velocity=args.target_velocity,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
