"""
OLSD Benchmark — Train RL baseline policies and generate expert trajectories.

Trains PPO/SAC via Stable Baselines3 on standard MuJoCo locomotion tasks,
saves checkpoints, and generates expert/medium trajectories for the dataset.

Usage:
    python -m olsd.benchmark.train_baseline --robot halfcheetah --algo ppo --timesteps 500000
    python -m olsd.benchmark.train_baseline --robot ant --algo sac --timesteps 1000000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np

from olsd.generation.mujoco_gen import GYMNASIUM_ROBOTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter presets (tuned for quick convergence on locomotion)
# ---------------------------------------------------------------------------

PPO_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
}

SAC_DEFAULTS = {
    "learning_rate": 3e-4,
    "buffer_size": 300_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "policy": "MlpPolicy",
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_policy(
    robot_id: str,
    algo: str = "ppo",
    total_timesteps: int = 500_000,
    output_dir: str | Path = "checkpoints",
    seed: int = 42,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
) -> Path:
    """
    Train an RL policy on a MuJoCo locomotion environment.

    Args:
        robot_id: Robot key from GYMNASIUM_ROBOTS
        algo: "ppo" or "sac"
        total_timesteps: Total training steps
        output_dir: Where to save checkpoints
        seed: Random seed
        eval_freq: Evaluate every N steps
        n_eval_episodes: Episodes per evaluation
        verbose: SB3 verbosity level

    Returns:
        Path to the best model checkpoint
    """
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    if robot_id not in GYMNASIUM_ROBOTS:
        raise ValueError(f"Unknown robot: {robot_id}. Available: {list(GYMNASIUM_ROBOTS.keys())}")

    env_id = GYMNASIUM_ROBOTS[robot_id]["env_id"]
    out_dir = Path(output_dir) / robot_id / algo
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {algo.upper()} on {env_id} for {total_timesteps:,} steps...")

    # Create environments
    env = Monitor(gym.make(env_id))
    eval_env = Monitor(gym.make(env_id))

    # Select algorithm
    if algo == "ppo":
        model = PPO(
            env=env,
            seed=seed,
            verbose=verbose,
            tensorboard_log=str(out_dir / "tb_logs"),
            **PPO_DEFAULTS,
        )
    elif algo == "sac":
        model = SAC(
            env=env,
            seed=seed,
            verbose=verbose,
            tensorboard_log=str(out_dir / "tb_logs"),
            **SAC_DEFAULTS,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use 'ppo' or 'sac'.")

    # Eval callback — saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save final model
    final_path = out_dir / "final_model"
    model.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    best_path = out_dir / "best" / "best_model.zip"
    if best_path.exists():
        logger.info(f"Best model at {best_path}")
        return best_path
    return Path(str(final_path) + ".zip")


# ---------------------------------------------------------------------------
# Generate expert trajectories from trained policy
# ---------------------------------------------------------------------------


def generate_expert_data(
    robot_id: str,
    model_path: str | Path,
    n_episodes: int = 500,
    output_dir: str | Path = "./data/olsd-expert",
    seed: int = 0,
) -> None:
    """Generate expert-quality trajectories using a trained policy."""
    from olsd.generation.mujoco_gen import generate_trajectories
    from olsd.pipeline.export import to_parquet
    from olsd.schema import TerrainType

    logger.info(f"Generating {n_episodes} expert episodes for {robot_id}...")

    episodes = generate_trajectories(
        robot_id=robot_id,
        n_episodes=n_episodes,
        max_steps=1000,
        policy=str(model_path),
        terrain=TerrainType.FLAT,
        seed=seed,
    )

    # Export
    out = Path(output_dir)
    to_parquet(episodes, out)
    logger.info(f"Expert data saved to {out}")


# ---------------------------------------------------------------------------
# Train all robots
# ---------------------------------------------------------------------------


def train_all(
    robots: list[str] | None = None,
    algo: str = "ppo",
    timesteps: int = 500_000,
    output_dir: str | Path = "checkpoints",
    seed: int = 42,
) -> dict[str, Path]:
    """Train policies for multiple robots, return dict of checkpoint paths."""
    robots = robots or list(GYMNASIUM_ROBOTS.keys())
    results = {}

    for robot_id in robots:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {robot_id} with {algo.upper()}")
        logger.info(f"{'='*60}")
        try:
            path = train_policy(
                robot_id=robot_id,
                algo=algo,
                total_timesteps=timesteps,
                output_dir=output_dir,
                seed=seed,
            )
            results[robot_id] = path
        except Exception as e:
            logger.error(f"Failed to train {robot_id}: {e}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="OLSD Baseline Training")
    parser.add_argument("--robot", "-r", type=str, default="halfcheetah",
                        help="Robot ID (halfcheetah, ant, walker2d, hopper)")
    parser.add_argument("--algo", "-a", type=str, default="ppo", choices=["ppo", "sac"],
                        help="RL algorithm")
    parser.add_argument("--timesteps", "-t", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--output", "-o", type=str, default="checkpoints",
                        help="Checkpoint output directory")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluate every N steps")
    parser.add_argument("--generate", "-g", action="store_true",
                        help="Generate expert data after training")
    parser.add_argument("--gen-episodes", type=int, default=500,
                        help="Number of expert episodes to generate")
    parser.add_argument("--gen-output", type=str, default="./data/olsd-expert",
                        help="Expert data output directory")
    parser.add_argument("--all-robots", action="store_true",
                        help="Train all available robots")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.all_robots:
        results = train_all(
            algo=args.algo,
            timesteps=args.timesteps,
            output_dir=args.output,
            seed=args.seed,
        )
        if args.generate:
            for robot_id, model_path in results.items():
                generate_expert_data(
                    robot_id=robot_id,
                    model_path=model_path,
                    n_episodes=args.gen_episodes,
                    output_dir=f"{args.gen_output}/{robot_id}",
                )
    else:
        model_path = train_policy(
            robot_id=args.robot,
            algo=args.algo,
            total_timesteps=args.timesteps,
            output_dir=args.output,
            seed=args.seed,
            eval_freq=args.eval_freq,
        )

        if args.generate:
            generate_expert_data(
                robot_id=args.robot,
                model_path=model_path,
                n_episodes=args.gen_episodes,
                output_dir=args.gen_output,
            )


if __name__ == "__main__":
    main()
