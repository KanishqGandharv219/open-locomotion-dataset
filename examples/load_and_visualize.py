"""
OLSD Example — Generate, visualize, and export a small dataset.

Usage:
    python examples/load_and_visualize.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from pathlib import Path

# 1. Generate trajectories
from olsd.generation.mujoco_gen import generate_trajectories

print("=" * 60)
print("🤖 OLSD — Generate & Visualize Example")
print("=" * 60)

print("\n📦 Generating 10 HalfCheetah episodes (random policy)...")
episodes = generate_trajectories(
    robot_id="halfcheetah",
    n_episodes=10,
    max_steps=200,
    policy="random",
    seed=42,
)
print(f"   Generated {len(episodes)} episodes, {sum(e.n_steps for e in episodes)} total steps")

# 2. Validate
from olsd.pipeline.validate import validate_dataset

print("\n🔍 Validating episodes...")
results = validate_dataset(episodes)
n_valid = sum(1 for r in results if r.valid)
print(f"   {n_valid}/{len(results)} episodes valid")

# 3. Compute metrics
from olsd.pipeline.metrics import compute_metrics

print("\n📐 Computing gait metrics for first episode...")
metrics = compute_metrics(episodes[0])
print(f"   Stride frequency: {metrics.stride_frequency:.2f} Hz")
print(f"   Energy per meter: {metrics.energy_per_meter:.2f} J/m")
print(f"   Smoothness index: {metrics.smoothness_index:.4f}")
print(f"   Duration: {metrics.duration_seconds:.2f} s")

# 4. Export to Parquet
from olsd.pipeline.export import to_parquet

output_dir = Path("data/sample")
print(f"\n💾 Exporting to Parquet at {output_dir}...")
to_parquet(episodes, output_dir)

# 5. Load back
import olsd

print(f"\n📥 Loading dataset from {output_dir}...")
dataset = olsd.load(str(output_dir))
summary = dataset.summary()
print(f"   Episodes: {summary['total_episodes']}")
print(f"   Steps: {summary['total_steps']}")
print(f"   Robots: {summary['robots']}")
print(f"   Success rate: {summary['success_rate']:.1%}")

# 6. Visualize (saves to file instead of showing)
from olsd.sdk.visualization import plot_trajectory, plot_phase_portrait, plot_rewards

viz_dir = Path("data/sample/visualizations")
viz_dir.mkdir(parents=True, exist_ok=True)

print(f"\n🎨 Generating visualizations in {viz_dir}/...")
plot_trajectory(dataset[0], show=False, save_path=viz_dir / "trajectory.png")
plot_phase_portrait(dataset[0], show=False, save_path=viz_dir / "phase_portrait.png")
plot_rewards(dataset[0], show=False, save_path=viz_dir / "rewards.png")

print("\n✅ Done! Check data/sample/ for outputs.")
print("=" * 60)
