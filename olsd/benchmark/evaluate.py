"""
OLSD Benchmark — Evaluate trained policies and compute standardized metrics.

Computes D4RL-style normalized scores + OLSD gait metrics.

Usage:
    python -m olsd.benchmark.evaluate --robot halfcheetah --model checkpoints/halfcheetah/ppo/best/best_model.zip
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np

from olsd.generation.mujoco_gen import GYMNASIUM_ROBOTS
from olsd.pipeline.metrics import compute_metrics
from olsd.pipeline.ingest import from_gymnasium

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# D4RL-style reference scores (approximate, for normalization)
# ---------------------------------------------------------------------------

REFERENCE_SCORES = {
    "halfcheetah": {"random": -280.0, "expert": 12135.0},
    "ant":         {"random": -70.0,  "expert": 6000.0},
    "walker2d":    {"random": 1.6,    "expert": 4592.3},
    "hopper":      {"random": -20.3,  "expert": 3234.3},
}


def normalized_score(robot_id: str, raw_score: float) -> float:
    """Compute D4RL-style normalized score: 0 = random, 100 = expert."""
    if robot_id not in REFERENCE_SCORES:
        return raw_score
    ref = REFERENCE_SCORES[robot_id]
    denom = ref["expert"] - ref["random"]
    if abs(denom) < 1e-8:
        return 0.0
    return (raw_score - ref["random"]) / denom * 100.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(
    robot_id: str,
    model_path: str | Path | None = None,
    n_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    deterministic: bool = True,
) -> dict:
    """
    Evaluate a trained policy on a MuJoCo locomotion environment.

    Args:
        robot_id: Robot key
        model_path: Path to SB3 checkpoint. If None, evaluates random policy.
        n_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        seed: Random seed
        deterministic: Use deterministic actions

    Returns:
        Dict with evaluation metrics
    """
    from stable_baselines3 import PPO, SAC

    if robot_id not in GYMNASIUM_ROBOTS:
        raise ValueError(f"Unknown robot: {robot_id}")

    env_id = GYMNASIUM_ROBOTS[robot_id]["env_id"]
    env = gym.make(env_id)

    # Load policy
    if model_path is not None:
        path = Path(model_path)
        try:
            model = PPO.load(path)
        except Exception:
            model = SAC.load(path)
        policy_name = path.stem
        logger.info(f"Evaluating trained policy: {path}")
    else:
        model = None
        policy_name = "random"
        logger.info(f"Evaluating random policy on {env_id}")

    # Run episodes
    returns = []
    lengths = []
    successes = []

    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        total_reward = 0.0
        done = False
        truncated = False
        t = 0

        while not (done or truncated) and t < max_steps:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            t += 1

        returns.append(total_reward)
        lengths.append(t)
        successes.append(not done)  # survived = success for locomotion

    env.close()

    # Compute statistics
    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)

    results = {
        "robot_id": robot_id,
        "policy": policy_name,
        "env_id": env_id,
        "n_episodes": n_episodes,
        "return_mean": float(returns_arr.mean()),
        "return_std": float(returns_arr.std()),
        "return_min": float(returns_arr.min()),
        "return_max": float(returns_arr.max()),
        "normalized_score": float(normalized_score(robot_id, returns_arr.mean())),
        "episode_length_mean": float(lengths_arr.mean()),
        "episode_length_std": float(lengths_arr.std()),
        "success_rate": float(np.mean(successes)),
    }

    # Also generate OLSD episodes for gait metrics
    logger.info("Computing gait metrics...")
    policy_fn = (lambda obs: model.predict(obs, deterministic=True)[0]) if model else None
    olsd_episodes = from_gymnasium(
        env_id=env_id,
        n_episodes=min(20, n_episodes),
        policy=policy_fn,
        max_steps=max_steps,
        seed=seed + 10000,
    )

    if olsd_episodes:
        gait_metrics = [compute_metrics(ep) for ep in olsd_episodes]
        results["gait_metrics"] = {
            "energy_per_meter_mean": float(np.mean([m.energy_per_meter for m in gait_metrics])),
            "stride_frequency_mean": float(np.mean([m.stride_frequency for m in gait_metrics])),
            "smoothness_mean": float(np.mean([m.smoothness_index for m in gait_metrics])),
            "mean_power_watts": float(np.mean([m.mean_power_watts for m in gait_metrics])),
        }

    return results


def evaluate_all(
    checkpoints_dir: str | Path = "checkpoints",
    n_episodes: int = 50,
    output_path: str | Path | None = None,
) -> list[dict]:
    """Evaluate all trained robots and print comparison table."""
    ckpt_dir = Path(checkpoints_dir)
    all_results = []

    # Evaluate random baseline for each robot
    for robot_id in GYMNASIUM_ROBOTS:
        logger.info(f"\n--- Evaluating RANDOM baseline: {robot_id} ---")
        result = evaluate_policy(robot_id, model_path=None, n_episodes=n_episodes)
        result["tier"] = "random"
        all_results.append(result)

    # Find and evaluate trained models
    for robot_dir in sorted(ckpt_dir.iterdir()) if ckpt_dir.exists() else []:
        if not robot_dir.is_dir():
            continue
        robot_id = robot_dir.name
        if robot_id not in GYMNASIUM_ROBOTS:
            continue

        for algo_dir in sorted(robot_dir.iterdir()):
            if not algo_dir.is_dir():
                continue

            best_model = algo_dir / "best" / "best_model.zip"
            final_model = algo_dir / "final_model.zip"
            model_path = best_model if best_model.exists() else final_model

            if not model_path.exists():
                continue

            logger.info(f"\n--- Evaluating {algo_dir.name.upper()}: {robot_id} ---")
            result = evaluate_policy(robot_id, model_path=model_path, n_episodes=n_episodes)
            result["tier"] = f"expert_{algo_dir.name}"
            all_results.append(result)

    # Print comparison table
    _print_results_table(all_results)

    # Save results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {out}")

    return all_results


def _print_results_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="🏆 OLSD Benchmark Results")
        table.add_column("Robot", style="cyan")
        table.add_column("Policy", style="magenta")
        table.add_column("Return", justify="right", style="green")
        table.add_column("± Std", justify="right")
        table.add_column("Norm. Score", justify="right", style="yellow")
        table.add_column("Ep. Length", justify="right")
        table.add_column("Success", justify="right")

        for r in sorted(results, key=lambda x: (x["robot_id"], x.get("tier", ""))):
            table.add_row(
                r["robot_id"],
                r.get("tier", r["policy"]),
                f"{r['return_mean']:.1f}",
                f"{r['return_std']:.1f}",
                f"{r['normalized_score']:.1f}",
                f"{r['episode_length_mean']:.0f}",
                f"{r['success_rate']:.1%}",
            )

        console.print(table)
    except ImportError:
        # Fallback without rich
        print(f"\n{'Robot':<15} {'Policy':<15} {'Return':>10} {'Norm.Score':>12} {'Success':>10}")
        print("-" * 65)
        for r in results:
            print(
                f"{r['robot_id']:<15} {r.get('tier', r['policy']):<15} "
                f"{r['return_mean']:>10.1f} {r['normalized_score']:>12.1f} "
                f"{r['success_rate']:>10.1%}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="OLSD Benchmark Evaluation")
    parser.add_argument("--robot", "-r", type=str, help="Robot ID to evaluate")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to SB3 model checkpoint")
    parser.add_argument("--episodes", "-n", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all robots (random + trained)")
    parser.add_argument("--checkpoints", type=str, default="checkpoints",
                        help="Checkpoints directory for --all mode")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results JSON to this path")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.all:
        evaluate_all(
            checkpoints_dir=args.checkpoints,
            n_episodes=args.episodes,
            output_path=args.output,
        )
    elif args.robot:
        result = evaluate_policy(
            robot_id=args.robot,
            model_path=args.model,
            n_episodes=args.episodes,
        )
        _print_results_table([result])

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
