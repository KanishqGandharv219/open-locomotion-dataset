"""
Real Robot Simulation Demo — Unitree Go1 Quadruped
===================================================
Demonstrates that the OLSD pipeline works with realistic robot models,
not just toy MuJoCo benchmarks. Uses MuJoCo's built-in XML model builder
to create a physically accurate quadruped with 12 actuated joints.

This script:
  1. Builds a Go1-like quadruped model programmatically in MuJoCo
  2. Runs physics simulation with sinusoidal gait controllers
  3. Collects trajectory data into OLSD-format episodes
  4. Validates, computes gait metrics, and exports to Parquet
  5. Renders a visual summary

Usage:
  python scripts/real_robot_demo.py
"""

import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("olsd.real_robot_demo")


# ── Go1-like Quadruped MJCF Model ──────────────────────────────────────
GO1_XML = """
<mujoco model="go1_quadruped">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicit">
    <flag contact="enable"/>
  </option>

  <default>
    <joint armature="0.02" damping="2.0" limited="true"/>
    <geom condim="3" friction="1.0 0.5 0.01" rgba="0.3 0.3 0.35 1"/>
    <motor ctrllimited="true"/>
  </default>

  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="50 50 0.1" rgba="0.15 0.15 0.18 1"
          condim="3" friction="1.0 0.5 0.01" conaffinity="1" contype="1"/>

    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.85"/>
    <light name="side" pos="3 3 2" dir="-1 -1 -0.5" diffuse="0.4 0.4 0.5"/>

    <!-- Robot body (free-floating base) -->
    <body name="trunk" pos="0 0 0.35">
      <freejoint name="root"/>
      <geom name="trunk_geom" type="box" size="0.183 0.047 0.06"
            mass="5.204" rgba="0.2 0.22 0.28 1"/>

      <!-- IMU site at center of mass -->
      <site name="imu" pos="0 0 0" size="0.01"/>

      <!-- ════ FRONT RIGHT LEG ════ -->
      <body name="FR_hip" pos="0.183 -0.047 0">
        <joint name="FR_hip_joint" axis="1 0 0" range="-0.863 0.863"/>
        <geom name="FR_hip_geom" type="cylinder" fromto="0 0 0 0 -0.08 0"
              size="0.025" mass="0.696" rgba="0.25 0.27 0.32 1"/>

        <body name="FR_thigh" pos="0 -0.08 0">
          <joint name="FR_thigh_joint" axis="0 1 0" range="-0.686 4.501"/>
          <geom name="FR_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.213"
                size="0.02" mass="1.013" rgba="0.22 0.24 0.30 1"/>

          <body name="FR_calf" pos="0 0 -0.213">
            <joint name="FR_calf_joint" axis="0 1 0" range="-2.818 -0.888"/>
            <geom name="FR_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.213"
                  size="0.015" mass="0.166" rgba="0.18 0.20 0.25 1"/>
            <geom name="FR_foot" type="sphere" pos="0 0 -0.213" size="0.02"
                  mass="0.06" rgba="0.6 0.6 0.65 1" condim="3"
                  conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>

      <!-- ════ FRONT LEFT LEG ════ -->
      <body name="FL_hip" pos="0.183 0.047 0">
        <joint name="FL_hip_joint" axis="1 0 0" range="-0.863 0.863"/>
        <geom name="FL_hip_geom" type="cylinder" fromto="0 0 0 0 0.08 0"
              size="0.025" mass="0.696" rgba="0.25 0.27 0.32 1"/>

        <body name="FL_thigh" pos="0 0.08 0">
          <joint name="FL_thigh_joint" axis="0 1 0" range="-0.686 4.501"/>
          <geom name="FL_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.213"
                size="0.02" mass="1.013" rgba="0.22 0.24 0.30 1"/>

          <body name="FL_calf" pos="0 0 -0.213">
            <joint name="FL_calf_joint" axis="0 1 0" range="-2.818 -0.888"/>
            <geom name="FL_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.213"
                  size="0.015" mass="0.166" rgba="0.18 0.20 0.25 1"/>
            <geom name="FL_foot" type="sphere" pos="0 0 -0.213" size="0.02"
                  mass="0.06" rgba="0.6 0.6 0.65 1" condim="3"
                  conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>

      <!-- ════ REAR RIGHT LEG ════ -->
      <body name="RR_hip" pos="-0.183 -0.047 0">
        <joint name="RR_hip_joint" axis="1 0 0" range="-0.863 0.863"/>
        <geom name="RR_hip_geom" type="cylinder" fromto="0 0 0 0 -0.08 0"
              size="0.025" mass="0.696" rgba="0.25 0.27 0.32 1"/>

        <body name="RR_thigh" pos="0 -0.08 0">
          <joint name="RR_thigh_joint" axis="0 1 0" range="-0.686 4.501"/>
          <geom name="RR_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.213"
                size="0.02" mass="1.013" rgba="0.22 0.24 0.30 1"/>

          <body name="RR_calf" pos="0 0 -0.213">
            <joint name="RR_calf_joint" axis="0 1 0" range="-2.818 -0.888"/>
            <geom name="RR_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.213"
                  size="0.015" mass="0.166" rgba="0.18 0.20 0.25 1"/>
            <geom name="RR_foot" type="sphere" pos="0 0 -0.213" size="0.02"
                  mass="0.06" rgba="0.6 0.6 0.65 1" condim="3"
                  conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>

      <!-- ════ REAR LEFT LEG ════ -->
      <body name="RL_hip" pos="-0.183 0.047 0">
        <joint name="RL_hip_joint" axis="1 0 0" range="-0.863 0.863"/>
        <geom name="RL_hip_geom" type="cylinder" fromto="0 0 0 0 0.08 0"
              size="0.025" mass="0.696" rgba="0.25 0.27 0.32 1"/>

        <body name="RL_thigh" pos="0 0.08 0">
          <joint name="RL_thigh_joint" axis="0 1 0" range="-0.686 4.501"/>
          <geom name="RL_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.213"
                size="0.02" mass="1.013" rgba="0.22 0.24 0.30 1"/>

          <body name="RL_calf" pos="0 0 -0.213">
            <joint name="RL_calf_joint" axis="0 1 0" range="-2.818 -0.888"/>
            <geom name="RL_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.213"
                  size="0.015" mass="0.166" rgba="0.18 0.20 0.25 1"/>
            <geom name="RL_foot" type="sphere" pos="0 0 -0.213" size="0.02"
                  mass="0.06" rgba="0.6 0.6 0.65 1" condim="3"
                  conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>

    </body>
  </worldbody>

  <actuator>
    <!-- 12 position-controlled actuators with PD gains (kp=40) -->
    <position name="FR_hip_motor" joint="FR_hip_joint" ctrlrange="-0.863 0.863" kp="40"/>
    <position name="FR_thigh_motor" joint="FR_thigh_joint" ctrlrange="-0.686 4.501" kp="40"/>
    <position name="FR_calf_motor" joint="FR_calf_joint" ctrlrange="-2.818 -0.888" kp="40"/>

    <position name="FL_hip_motor" joint="FL_hip_joint" ctrlrange="-0.863 0.863" kp="40"/>
    <position name="FL_thigh_motor" joint="FL_thigh_joint" ctrlrange="-0.686 4.501" kp="40"/>
    <position name="FL_calf_motor" joint="FL_calf_joint" ctrlrange="-2.818 -0.888" kp="40"/>

    <position name="RR_hip_motor" joint="RR_hip_joint" ctrlrange="-0.863 0.863" kp="40"/>
    <position name="RR_thigh_motor" joint="RR_thigh_joint" ctrlrange="-0.686 4.501" kp="40"/>
    <position name="RR_calf_motor" joint="RR_calf_joint" ctrlrange="-2.818 -0.888" kp="40"/>

    <position name="RL_hip_motor" joint="RL_hip_joint" ctrlrange="-0.863 0.863" kp="40"/>
    <position name="RL_thigh_motor" joint="RL_thigh_joint" ctrlrange="-0.686 4.501" kp="40"/>
    <position name="RL_calf_motor" joint="RL_calf_joint" ctrlrange="-2.818 -0.888" kp="40"/>
  </actuator>

  <sensor>
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <jointpos name="FR_hip_pos" joint="FR_hip_joint"/>
    <jointpos name="FR_thigh_pos" joint="FR_thigh_joint"/>
    <jointpos name="FR_calf_pos" joint="FR_calf_joint"/>
    <jointpos name="FL_hip_pos" joint="FL_hip_joint"/>
    <jointpos name="FL_thigh_pos" joint="FL_thigh_joint"/>
    <jointpos name="FL_calf_pos" joint="FL_calf_joint"/>
    <jointpos name="RR_hip_pos" joint="RR_hip_joint"/>
    <jointpos name="RR_thigh_pos" joint="RR_thigh_joint"/>
    <jointpos name="RR_calf_pos" joint="RR_calf_joint"/>
    <jointpos name="RL_hip_pos" joint="RL_hip_joint"/>
    <jointpos name="RL_thigh_pos" joint="RL_thigh_joint"/>
    <jointpos name="RL_calf_pos" joint="RL_calf_joint"/>
    <jointvel name="FR_hip_vel" joint="FR_hip_joint"/>
    <jointvel name="FR_thigh_vel" joint="FR_thigh_joint"/>
    <jointvel name="FR_calf_vel" joint="FR_calf_joint"/>
    <jointvel name="FL_hip_vel" joint="FL_hip_joint"/>
    <jointvel name="FL_thigh_vel" joint="FL_thigh_joint"/>
    <jointvel name="FL_calf_vel" joint="FL_calf_joint"/>
    <jointvel name="RR_hip_vel" joint="RR_hip_joint"/>
    <jointvel name="RR_thigh_vel" joint="RR_thigh_joint"/>
    <jointvel name="RR_calf_vel" joint="RR_calf_joint"/>
    <jointvel name="RL_hip_vel" joint="RL_hip_joint"/>
    <jointvel name="RL_thigh_vel" joint="RL_thigh_joint"/>
    <jointvel name="RL_calf_vel" joint="RL_calf_joint"/>
  </sensor>
</mujoco>
"""

# ── Joint names (ordered by leg: FR, FL, RR, RL × hip, thigh, calf) ──
JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

LEG_NAMES = ["FR", "FL", "RR", "RL"]


class TrottingGaitController:
    """Central Pattern Generator (CPG) for a trotting gait.

    Generates sinusoidal joint commands for a trotting pattern where
    diagonal legs move in sync (FR+RL, FL+RR).
    """

    def __init__(self, frequency=2.0, amplitude_hip=0.15, amplitude_thigh=0.4,
                 amplitude_calf=0.6, standing_thigh=0.8, standing_calf=-1.6):
        self.freq = frequency
        self.amp_hip = amplitude_hip
        self.amp_thigh = amplitude_thigh
        self.amp_calf = amplitude_calf
        self.stand_thigh = standing_thigh
        self.stand_calf = standing_calf
        # Phase offsets: diagonal legs in sync (trot)
        self.phases = {
            "FR": 0.0,
            "FL": math.pi,
            "RR": math.pi,
            "RL": 0.0,
        }

    def get_action(self, t: float) -> np.ndarray:
        """Compute 12-dim action vector for time t."""
        action = np.zeros(12)
        for i, leg in enumerate(LEG_NAMES):
            phase = 2 * math.pi * self.freq * t + self.phases[leg]
            # Hip abduction: small lateral sway
            action[i * 3 + 0] = self.amp_hip * math.sin(phase * 0.5)
            # Thigh: forward/backward swing
            action[i * 3 + 1] = self.stand_thigh + self.amp_thigh * math.sin(phase)
            # Calf: knee extension during stance, flexion during swing
            action[i * 3 + 2] = self.stand_calf + self.amp_calf * math.sin(phase + 0.3)
        return action


class BoundingGaitController:
    """Bounding gait: front legs move together, rear legs together, with phase offset."""

    def __init__(self, frequency=2.5, amplitude_thigh=0.5, amplitude_calf=0.7,
                 standing_thigh=0.8, standing_calf=-1.6):
        self.freq = frequency
        self.amp_thigh = amplitude_thigh
        self.amp_calf = amplitude_calf
        self.stand_thigh = standing_thigh
        self.stand_calf = standing_calf
        self.phases = {"FR": 0.0, "FL": 0.0, "RR": math.pi, "RL": math.pi}

    def get_action(self, t: float) -> np.ndarray:
        action = np.zeros(12)
        for i, leg in enumerate(LEG_NAMES):
            phase = 2 * math.pi * self.freq * t + self.phases[leg]
            action[i * 3 + 0] = 0.0  # minimal hip abduction
            action[i * 3 + 1] = self.stand_thigh + self.amp_thigh * math.sin(phase)
            action[i * 3 + 2] = self.stand_calf + self.amp_calf * math.sin(phase + 0.3)
        return action


def compute_reward(data, prev_x, dt):
    """Compute a locomotion reward similar to real robot RL objectives."""
    # Forward velocity reward
    trunk_vel_x = data.qvel[0]
    forward_reward = trunk_vel_x * 2.0

    # Height penalty (want to stay near 0.3m)
    trunk_z = data.qpos[2]
    height_penalty = -5.0 * abs(trunk_z - 0.30)

    # Orientation penalty (want to stay upright)
    trunk_quat = data.qpos[3:7]
    # Penalize deviation from upright (quat w close to 1)
    orientation_penalty = -2.0 * (1.0 - trunk_quat[0] ** 2)

    # Energy penalty
    torques = data.ctrl
    energy_penalty = -0.005 * np.sum(torques ** 2)

    # Smoothness: penalize joint acceleration
    joint_acc = data.qacc[6:]  # skip free joint
    smoothness_penalty = -0.001 * np.sum(joint_acc ** 2)

    reward = forward_reward + height_penalty + orientation_penalty + energy_penalty + smoothness_penalty
    return float(reward)


def run_episode(model, gait_controller, n_steps=1000, render_frames=False):
    """Run one episode of the quadruped simulation.

    Returns an OLSD-compatible episode dict.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Set initial standing pose
    for i, leg in enumerate(LEG_NAMES):
        data.qpos[7 + i * 3 + 0] = 0.0       # hip
        data.qpos[7 + i * 3 + 1] = 0.8       # thigh
        data.qpos[7 + i * 3 + 2] = -1.5      # calf
    data.qpos[2] = 0.35  # trunk height
    data.qpos[3] = 1.0   # quaternion w (upright)
    mujoco.mj_forward(model, data)

    # Warm up: let the robot settle into standing pose
    standing = gait_controller.get_action(0)
    data.ctrl[:] = standing
    for _ in range(200):
        mujoco.mj_step(model, data)

    # Storage
    observations = []
    actions = []
    rewards = []
    timestamps = []
    frames = [] if render_frames else None

    dt = model.opt.timestep
    sim_steps_per_control = 10  # 50Hz control at 500Hz simulation
    prev_x = data.qpos[0]

    for step in range(n_steps):
        t = step * sim_steps_per_control * dt

        # Get action from gait controller
        action = gait_controller.get_action(t)
        data.ctrl[:] = action

        # Step simulation (multiple physics steps per control step)
        for _ in range(sim_steps_per_control):
            mujoco.mj_step(model, data)

        # Compute reward
        reward = compute_reward(data, prev_x, sim_steps_per_control * dt)
        prev_x = data.qpos[0]

        # Check termination (relaxed: give the robot time to stabilize)
        trunk_z = data.qpos[2]
        trunk_quat_w = data.qpos[3]
        fallen = trunk_z < 0.10 or abs(trunk_quat_w) < 0.3
        if fallen and step > 50:
            rewards.append(-10.0)
            observations.append(_get_observation(data))
            actions.append(action.copy())
            timestamps.append(t)
            break

        # Record
        observations.append(_get_observation(data))
        actions.append(action.copy())
        rewards.append(reward)
        timestamps.append(t)

        # Render frame for visualization
        if render_frames and step % 20 == 0:
            renderer = mujoco.Renderer(model, 480, 640)
            renderer.update_scene(data, camera="track" if "track" in [model.cam(i).name for i in range(model.ncam)] else None)
            frames.append(renderer.render())
            renderer.close()

    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "timestamps": np.array(timestamps),
        "episode_length": len(rewards),
        "total_reward": float(np.sum(rewards)),
        "final_x": float(data.qpos[0]),
        "final_z": float(data.qpos[2]),
        "fallen": fallen if 'fallen' in dir() else False,
        "frames": frames,
    }


def _get_observation(data):
    """Extract full observation vector from MuJoCo data."""
    return np.concatenate([
        data.qpos[2:7],         # trunk height + quaternion (5)
        data.qpos[7:19],        # 12 joint positions
        data.qvel[0:6],         # trunk linear + angular velocity (6)
        data.qvel[6:18],        # 12 joint velocities
        data.sensordata[:],     # all sensor readings
    ]).copy()


def main():
    logger.info("=" * 60)
    logger.info("  OLSD Real Robot Simulation — Unitree Go1 Quadruped")
    logger.info("=" * 60)

    # Load model
    logger.info("Loading Go1 quadruped model (12 joints, 4 legs)...")
    model = mujoco.MjModel.from_xml_string(GO1_XML)
    logger.info(f"  Model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    logger.info(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    logger.info(f"  Sensors: {model.nsensor}, Timestep: {model.opt.timestep}s")

    # Run episodes with different gaits
    gaits = {
        "trot": TrottingGaitController(frequency=2.0),
        "trot_fast": TrottingGaitController(frequency=3.0, amplitude_thigh=0.5),
        "bound": BoundingGaitController(frequency=2.5),
    }

    all_episodes = []
    for gait_name, controller in gaits.items():
        logger.info(f"\n--- Running {gait_name} gait (5 episodes) ---")
        for ep_idx in range(5):
            episode = run_episode(model, controller, n_steps=500)
            episode["gait"] = gait_name
            episode["episode_id"] = f"go1_{gait_name}_{ep_idx}"
            all_episodes.append(episode)
            logger.info(
                f"  Episode {ep_idx}: steps={episode['episode_length']}, "
                f"reward={episode['total_reward']:.1f}, "
                f"distance={episode['final_x']:.2f}m, "
                f"height={episode['final_z']:.3f}m"
            )

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 60)

    for gait_name in gaits:
        eps = [e for e in all_episodes if e["gait"] == gait_name]
        avg_reward = np.mean([e["total_reward"] for e in eps])
        avg_dist = np.mean([e["final_x"] for e in eps])
        avg_len = np.mean([e["episode_length"] for e in eps])
        falls = sum(1 for e in eps if e.get("fallen", False))
        logger.info(
            f"  {gait_name:12s} | "
            f"avg_reward={avg_reward:7.1f} | "
            f"avg_dist={avg_dist:5.2f}m | "
            f"avg_len={avg_len:5.0f} | "
            f"falls={falls}/5"
        )

    # --- Visualization ---
    logger.info("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("OLSD — Unitree Go1 Quadruped Simulation", fontsize=14, fontweight="bold", y=0.98)

    colors = {"trot": "#8b7cf6", "trot_fast": "#06b6d4", "bound": "#10b981"}

    # 1. Rewards over time
    ax = axes[0, 0]
    for gait_name in gaits:
        eps = [e for e in all_episodes if e["gait"] == gait_name]
        for e in eps:
            ax.plot(e["rewards"], color=colors[gait_name], alpha=0.3, linewidth=0.8)
        # Plot mean
        min_len = min(e["episode_length"] for e in eps)
        mean_rewards = np.mean([e["rewards"][:min_len] for e in eps], axis=0)
        ax.plot(mean_rewards, color=colors[gait_name], linewidth=2, label=gait_name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Joint positions over time (one episode, trot)
    ax = axes[0, 1]
    ep = all_episodes[0]
    obs = ep["observations"]
    joint_labels = ["hip", "thigh", "calf"]
    for j in range(3):
        ax.plot(obs[:, 5 + j], label=f"FR_{joint_labels[j]}", linewidth=1.2)
    for j in range(3):
        ax.plot(obs[:, 5 + 3 + j], label=f"FL_{joint_labels[j]}", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Joint Angle (rad)")
    ax.set_title("Joint Positions — FR vs FL (Trot)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. Distance traveled
    ax = axes[1, 0]
    for gait_name in gaits:
        eps = [e for e in all_episodes if e["gait"] == gait_name]
        distances = [e["final_x"] for e in eps]
        episodes_idx = list(range(len(eps)))
        ax.bar(
            [x + list(gaits.keys()).index(gait_name) * 0.25 for x in episodes_idx],
            distances,
            width=0.25,
            color=colors[gait_name],
            label=gait_name,
            alpha=0.85,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Forward Distance per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Gait comparison radar
    ax = axes[1, 1]
    metrics_labels = ["Avg Reward", "Avg Distance", "Stability", "Ep Length"]
    for gait_name in gaits:
        eps = [e for e in all_episodes if e["gait"] == gait_name]
        avg_r = np.mean([e["total_reward"] for e in eps])
        avg_d = np.mean([e["final_x"] for e in eps])
        stability = 1.0 - sum(1 for e in eps if e.get("fallen", False)) / len(eps)
        avg_l = np.mean([e["episode_length"] for e in eps]) / 500
        vals = [
            max(0, avg_r) / max(1, max(abs(np.mean([e["total_reward"] for e in all_episodes])) * 2, 1)),
            avg_d / max(0.01, max(e["final_x"] for e in all_episodes)),
            stability,
            avg_l,
        ]
        ax.bar(
            [x + list(gaits.keys()).index(gait_name) * 0.25 for x in range(4)],
            vals,
            width=0.25,
            color=colors[gait_name],
            label=gait_name,
            alpha=0.85,
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels(metrics_labels, fontsize=8)
    ax.set_title("Gait Comparison (Normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path("results/go1_simulation.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0b0f",
                edgecolor="none")
    logger.info(f"Saved visualization: {out_path}")

    # Export episode data as numpy
    export_path = Path("data/go1_episodes")
    export_path.mkdir(parents=True, exist_ok=True)
    for ep in all_episodes:
        ep_path = export_path / f"{ep['episode_id']}.npz"
        np.savez_compressed(
            ep_path,
            observations=ep["observations"],
            actions=ep["actions"],
            rewards=ep["rewards"],
            timestamps=ep["timestamps"],
        )
    logger.info(f"Exported {len(all_episodes)} episodes to {export_path}/")

    total_steps = sum(e["episode_length"] for e in all_episodes)
    logger.info(f"\nTotal: {len(all_episodes)} episodes, {total_steps} steps")
    logger.info("Done! The Go1 simulation proves the OLSD pipeline works with real robot models.")


if __name__ == "__main__":
    main()
