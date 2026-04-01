"""Re-benchmark the new velocity-tracking Go1 policies for v2.0 release."""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from olsd.sim2real.go1_env import Go1SimEnv
from olsd.schema import TerrainType
from olsd.sim2real.go1_compare import (
    resolve_go1_sim_params,
    default_go1_terrain_params,
    summarize_episode_records,
    SHARED_GO1_METRICS,
)

SIM2REAL_CONFIG = "configs/sim2real/go1.yaml"
POLICIES = {
    "flat": "checkpoints/go1-veltrack-v1/flat/go1_ppo",
    "stairs": "checkpoints/go1-veltrack-v1/stairs/selected_model",
}
N_EVAL = 20
HORIZON = 1000
SEED = 0
TARGET_VEL = 0.5

sim_params = resolve_go1_sim_params(SIM2REAL_CONFIG)

def evaluate(policy_path, terrain, label):
    from stable_baselines3 import PPO
    model = PPO.load(str(policy_path), device="cpu")
    terrain_params = default_go1_terrain_params(terrain)
    records = []
    for ep in range(N_EVAL):
        env = Go1SimEnv(
            terrain_type=terrain,
            terrain_params=terrain_params,
            sim_params=sim_params,
            max_steps=HORIZON,
            target_velocity=TARGET_VEL,
            randomize_velocity=False,
        )
        obs, _ = env.reset(seed=SEED + ep)
        steps = 0
        terminated = truncated = False
        vel_sum = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            vel_sum += info["x_velocity"]
            steps += 1
        env.close()
        records.append({
            "success": not terminated,
            "fall": terminated,
            "episode_length": steps,
            "forward_velocity_mean": vel_sum / max(steps, 1),
        })
    summary = summarize_episode_records(records)
    print(f"\n{label} on {terrain.value}:")
    print(f"  success_rate:       {summary['success_rate']}")
    print(f"  episode_length:     {summary['episode_length_mean']}")
    print(f"  forward_velocity:   {summary['forward_velocity_mean']:.4f} m/s")
    print(f"  fall_count:         {summary['fall_count']}")
    return summary

results = {}

# Evaluate flat
print("="*60)
print("EVALUATING NEW VELOCITY-TRACKING POLICIES (v2.0)")
print("="*60)

results["olsd_v2_veltrack"] = {}
for terrain_name, policy_path in POLICIES.items():
    terrain = TerrainType(terrain_name)
    summary = evaluate(policy_path, terrain, "OLSD v2 veltrack")
    results["olsd_v2_veltrack"][terrain_name] = summary

# Build the report
report = {
    "schema_version": "2.0",
    "robot_id": "unitree_go1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "reward_version": "velocity_tracking_v1",
    "protocol": {
        "terrains": ["flat", "stairs"],
        "slope_status": "in_progress",
        "n_eval_episodes": N_EVAL,
        "horizon": HORIZON,
        "seed": SEED,
        "target_velocity_ms": TARGET_VEL,
        "shared_metrics": SHARED_GO1_METRICS,
        "reward_policy": "excluded_from_comparison",
    },
    "baselines": results,
}

out_path = Path("results/go1_head_to_head.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved to {out_path}")
print("Done!")
