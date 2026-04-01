"""Quick verification: is the flat Go1 policy *actually* walking?"""
import numpy as np
from olsd.sim2real.go1_env import Go1SimEnv
from olsd.schema import TerrainType

try:
    from stable_baselines3 import PPO
except ImportError:
    raise SystemExit("stable-baselines3 not installed")

POLICY_PATH = "checkpoints/go1-veltrack-v1/flat/go1_ppo"
print(f"Loading policy from: {POLICY_PATH}")
model = PPO.load(POLICY_PATH, device="cpu")

env = Go1SimEnv(
    terrain_type=TerrainType.FLAT,
    target_velocity=0.5,
    randomize_velocity=False,
)
obs, _ = env.reset(seed=42)

vels = []
heights = []
x_positions = []
steps = 0
terminated = False
truncated = False
x_start = float(env.data.qpos[0])

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    vels.append(info["x_velocity"])
    heights.append(info["height"])
    x_positions.append(float(env.data.qpos[0]))
    steps += 1

x_end = float(env.data.qpos[0])

print(f"\n{'='*50}")
print(f"FLAT POLICY VERIFICATION")
print(f"{'='*50}")
print(f"Total steps:           {steps}")
print(f"Fell (terminated):     {terminated}")
print(f"X start:               {x_start:.4f} m")
print(f"X end:                 {x_end:.4f} m")
print(f"Total X displacement:  {x_end - x_start:.4f} m")
print(f"Mean forward velocity: {np.mean(vels):.4f} m/s")
print(f"Std forward velocity:  {np.std(vels):.4f} m/s")
print(f"Max forward velocity:  {np.max(vels):.4f} m/s")
print(f"Min forward velocity:  {np.min(vels):.4f} m/s")
print(f"Mean height:           {np.mean(heights):.4f} m")
print(f"Min height:            {np.min(heights):.4f} m")
print(f"{'='*50}")

if abs(x_end - x_start) < 0.5:
    print("VERDICT: STANDING STILL. The policy is NOT walking.")
elif abs(x_end - x_start) < 2.0:
    print("VERDICT: SHUFFLING. Minor movement but not real locomotion.")
elif np.mean(vels) > 0.2:
    print(f"VERDICT: WALKING! Covered {x_end - x_start:.1f}m at {np.mean(vels):.2f} m/s avg.")
else:
    print("VERDICT: INCONCLUSIVE. Check the numbers above.")

env.close()
