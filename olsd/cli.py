"""
OLSD CLI — Command-line interface for dataset management.

Usage:
    olsd info <path>                    # Show dataset stats
    olsd generate --robot go1 --episodes 100
    olsd validate <path>                # Validate a dataset/submission
    olsd export --format parquet --output ./out
    olsd upload --repo <hf-repo-id>     # Push to Hugging Face
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger("olsd")


@click.group()
@click.version_option(version="0.1.0", prog_name="olsd")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """OLSD — Open Locomotion Skills Dataset CLI"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@main.command()
@click.argument("path", type=click.Path(exists=True))
def info(path: str):
    """Show dataset statistics."""
    from olsd.sdk.loader import load

    dataset = load(path)
    summary = dataset.summary()

    table = Table(title=f"📊 OLSD Dataset: {path}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Episodes", str(summary["total_episodes"]))
    table.add_row("Total Steps", str(summary["total_steps"]))
    table.add_row("Robots", ", ".join(summary["robots"]))
    table.add_row("Morphologies", ", ".join(summary["morphologies"]))
    table.add_row("Terrains", ", ".join(summary["terrains"]))
    table.add_row("Sources", ", ".join(summary["sources"]))
    table.add_row("Success Rate", f"{summary['success_rate']:.1%}")

    console.print(table)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@main.command()
@click.option("--robot", "-r", multiple=True, default=["halfcheetah"],
              help="Robot ID(s) to generate for")
@click.option("--episodes", "-n", default=100, help="Episodes per robot/config combo")
@click.option("--max-steps", default=1000, help="Max steps per episode")
@click.option("--policy", "-p", default="random",
              help="Policy: 'random', 'heuristic', or path to SB3 checkpoint")
@click.option("--terrain", "-t", default="flat", help="Terrain type")
@click.option("--domain-random", is_flag=True, help="Enable domain randomization")
@click.option("--output", "-o", default="./data/generated", help="Output directory")
@click.option("--format", "fmt", default="parquet",
              type=click.Choice(["parquet", "hdf5", "json"]))
@click.option("--seed", default=42, help="Random seed")
@click.option("--configs-dir", default="configs/robots", help="Robot configs directory")
def generate(robot, episodes, max_steps, policy, terrain, domain_random, output, fmt, seed, configs_dir):
    """Generate synthetic locomotion trajectories."""
    from olsd.generation.mujoco_gen import generate_trajectories
    from olsd.pipeline.export import to_hdf5, to_json, to_parquet
    from olsd.schema import TerrainType

    terrain_type = TerrainType(terrain)
    all_episodes = []

    for robot_id in robot:
        console.print(f"🤖 Generating {episodes} episodes for [cyan]{robot_id}[/cyan]...")
        eps = generate_trajectories(
            robot_id=robot_id,
            n_episodes=episodes,
            max_steps=max_steps,
            policy=policy,
            terrain=terrain_type,
            domain_randomization=domain_random,
            seed=seed,
            configs_dir=configs_dir,
        )
        all_episodes.extend(eps)
        console.print(f"  ✓ Generated {len(eps)} episodes ({sum(e.n_steps for e in eps)} steps)")

    console.print(f"\n📦 Exporting {len(all_episodes)} episodes as {fmt}...")

    output_path = Path(output)
    if fmt == "parquet":
        to_parquet(all_episodes, output_path)
    elif fmt == "hdf5":
        to_hdf5(all_episodes, output_path / "dataset.hdf5")
    elif fmt == "json":
        to_json(all_episodes, output_path / "dataset.json")

    console.print(f"✅ Dataset saved to [green]{output_path}[/green]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@main.command()
@click.argument("path", type=click.Path(exists=True))
def validate(path: str):
    """Validate a dataset or submission."""
    from olsd.pipeline.validate import validate_dataset
    from olsd.sdk.loader import load

    console.print(f"🔍 Validating dataset at [cyan]{path}[/cyan]...")

    dataset = load(path)
    results = validate_dataset(dataset.episodes)

    n_valid = sum(1 for r in results if r.valid)
    n_invalid = len(results) - n_valid

    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total Episodes", str(len(results)))
    table.add_row("Valid", f"[green]{n_valid}[/green]")
    table.add_row("Invalid", f"[red]{n_invalid}[/red]" if n_invalid else f"[green]{n_invalid}[/green]")

    console.print(table)

    # Show details for invalid episodes
    if n_invalid > 0:
        console.print("\n[red]Invalid episodes:[/red]")
        for r in results:
            if not r.valid:
                console.print(f"  Episode {r.episode_id}:")
                for e in r.errors:
                    console.print(f"    [red]ERROR[/red]: {e}")
                for w in r.warnings:
                    console.print(f"    [yellow]WARN[/yellow]: {w}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--format", "fmt", required=True,
              type=click.Choice(["parquet", "hdf5", "json", "hf"]))
@click.option("--output", "-o", required=True, help="Output path")
def export(input_path: str, fmt: str, output: str):
    """Export dataset to a different format."""
    from olsd.pipeline.export import to_hdf5, to_hf_dataset, to_json, to_parquet
    from olsd.sdk.loader import load

    console.print(f"📥 Loading from [cyan]{input_path}[/cyan]...")
    dataset = load(input_path)

    console.print(f"📦 Exporting {len(dataset)} episodes to {fmt}...")

    if fmt == "parquet":
        to_parquet(dataset.episodes, output)
    elif fmt == "hdf5":
        to_hdf5(dataset.episodes, output)
    elif fmt == "json":
        to_json(dataset.episodes, output)
    elif fmt == "hf":
        hf_ds = to_hf_dataset(dataset.episodes)
        hf_ds.save_to_disk(output)

    console.print(f"✅ Exported to [green]{output}[/green]")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Save metrics to JSON file")
def metrics(path: str, output: str | None):
    """Compute gait metrics for a dataset."""
    from olsd.pipeline.metrics import compute_dataset_metrics, compute_metrics
    from olsd.sdk.loader import load

    dataset = load(path)
    console.print(f"📐 Computing metrics for {len(dataset)} episodes...")

    summary = compute_dataset_metrics(dataset.episodes)

    table = Table(title="Dataset Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in summary.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)

    if output:
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"💾 Metrics saved to [green]{output}[/green]")


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--repo", "-r", required=True, help="Hugging Face repo ID (e.g. org/dataset)")
@click.option("--private", is_flag=True, help="Create as private dataset")
def upload(path: str, repo: str, private: bool):
    """Upload dataset to Hugging Face Hub."""
    from olsd.pipeline.export import to_hf_dataset
    from olsd.sdk.loader import load

    console.print(f"📤 Uploading [cyan]{path}[/cyan] to [green]{repo}[/green]...")

    dataset = load(path)
    hf_ds = to_hf_dataset(dataset.episodes)

    hf_ds.push_to_hub(repo, private=private)
    console.print(f"✅ Uploaded to https://huggingface.co/datasets/{repo}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@main.command()
@click.option("--robot", "-r", default="halfcheetah", help="Robot ID")
@click.option("--algo", "-a", default="ppo", type=click.Choice(["ppo", "sac"]),
              help="RL algorithm")
@click.option("--timesteps", "-t", default=500_000, help="Total training timesteps")
@click.option("--output", "-o", default="checkpoints", help="Checkpoint directory")
@click.option("--seed", default=42, help="Random seed")
@click.option("--generate", "-g", is_flag=True, help="Generate expert data after training")
@click.option("--all-robots", is_flag=True, help="Train all robots")
def train(robot, algo, timesteps, output, seed, generate, all_robots):
    """Train RL baseline policies."""
    from olsd.benchmark.train_baseline import (
        generate_expert_data,
        train_all,
        train_policy,
    )

    if all_robots:
        console.print(f"🏋️ Training [cyan]{algo.upper()}[/cyan] on all robots for {timesteps:,} steps...")
        results = train_all(algo=algo, timesteps=timesteps, output_dir=output, seed=seed)
        for rid, path in results.items():
            console.print(f"  ✓ {rid}: {path}")
            if generate:
                generate_expert_data(rid, path, n_episodes=500, output_dir=f"./data/olsd-expert/{rid}")
    else:
        console.print(f"🏋️ Training [cyan]{algo.upper()}[/cyan] on [green]{robot}[/green] for {timesteps:,} steps...")
        path = train_policy(robot_id=robot, algo=algo, total_timesteps=timesteps, output_dir=output, seed=seed)
        console.print(f"✅ Model saved to [green]{path}[/green]")
        if generate:
            generate_expert_data(robot, path, n_episodes=500, output_dir=f"./data/olsd-expert/{robot}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@main.command("eval")
@click.option("--robot", "-r", default=None, help="Robot ID to evaluate")
@click.option("--model", "-m", default=None, help="Path to SB3 checkpoint")
@click.option("--episodes", "-n", default=50, help="Evaluation episodes")
@click.option("--all", "eval_all", is_flag=True, help="Evaluate all robots")
@click.option("--checkpoints", default="checkpoints", help="Checkpoints dir for --all")
@click.option("--output", "-o", default=None, help="Save results JSON")
def evaluate(robot, model, episodes, eval_all, checkpoints, output):
    """Evaluate trained policies with D4RL-style scores."""
    from olsd.benchmark.evaluate import evaluate_all, evaluate_policy, _print_results_table

    if eval_all:
        evaluate_all(checkpoints_dir=checkpoints, n_episodes=episodes, output_path=output)
    elif robot:
        result = evaluate_policy(robot_id=robot, model_path=model, n_episodes=episodes)
        _print_results_table([result])
        if output:
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
    else:
        console.print("[red]Specify --robot or --all[/red]")


if __name__ == "__main__":
    main()
