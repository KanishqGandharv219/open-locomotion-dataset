"""Reusable Go1 MuJoCo environment and rollout helpers."""

from __future__ import annotations

from collections import deque

import numpy as np

from olsd.pipeline.ingest import load_robot_by_id
from olsd.schema import (
    Action,
    ControlMode,
    DataSource,
    Episode,
    EpisodeMetadata,
    Observation,
    RobotSpec,
    Step,
    TerrainSpec,
    TerrainType,
)
from olsd.sim2real.system_id import SimParams
from olsd.sim2real.terrain import generate_terrain_xml

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    spaces = None

try:
    import mujoco
except ImportError:  # pragma: no cover - optional dependency
    mujoco = None


GO1_JOINT_LIMITS = np.array(
    [
        [-0.863, 0.863],
        [-0.686, 4.501],
        [-2.818, -0.888],
        [-0.863, 0.863],
        [-0.686, 4.501],
        [-2.818, -0.888],
        [-0.863, 0.863],
        [-0.686, 4.501],
        [-2.818, -0.888],
        [-0.863, 0.863],
        [-0.686, 4.501],
        [-2.818, -0.888],
    ],
    dtype=np.float32,
)
GO1_STANDING_POSE = np.array(
    [0.0, 0.8, -1.5] * 4,
    dtype=np.float32,
)
GO1_TARGET_CLEARANCE = 0.30
GO1_POLICY_ACTION_RANGE = (
    np.minimum(
        GO1_STANDING_POSE - GO1_JOINT_LIMITS[:, 0],
        GO1_JOINT_LIMITS[:, 1] - GO1_STANDING_POSE,
    )
    * 0.35
).astype(np.float32)
LEG_NAMES = ["FR", "FL", "RR", "RL"]

GO1_MODEL_TEMPLATE = """
<mujoco model="go1_quadruped">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicit">
    <flag contact="enable"/>
  </option>
  <default>
    <joint armature="{armature}" damping="{damping}" limited="true"/>
    <geom condim="3" friction="{friction} 0.5 0.01" rgba="0.3 0.3 0.35 1"/>
    <motor ctrllimited="true"/>
  </default>
  <worldbody>
    {terrain_xml}
    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.85"/>
    <body name="trunk" pos="0 0 0.35">
      <freejoint name="root"/>
      <geom name="trunk_geom" type="box" size="0.183 0.047 0.06" mass="5.204" rgba="0.2 0.22 0.28 1"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <body name="FR_hip" pos="0.183 -0.047 0">
        <joint name="FR_hip_joint" axis="1 0 0" range="-0.863 0.863"/>
        <geom name="FR_hip_geom" type="cylinder" fromto="0 0 0 0 -0.08 0" size="0.025" mass="0.696"/>
        <body name="FR_thigh" pos="0 -0.08 0">
          <joint name="FR_thigh_joint" axis="0 1 0" range="-0.686 4.501"/>
          <geom name="FR_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.213" size="0.02" mass="1.013"/>
          <body name="FR_calf" pos="0 0 -0.213">
            <joint name="FR_calf_joint" axis="0 1 0" range="-2.818 -0.888"/>
            <geom name="FR_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.213" size="0.015" mass="0.166"/>
            <geom name="FR_foot" type="sphere" pos="0 0 -0.213" size="0.02" mass="0.06" condim="3" conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>
      <body name="FL_hip" pos="0.183 0.047 0">
        <joint name="FL_hip_joint" axis="1 0 0" range="-0.863 0.863"/>
        <geom name="FL_hip_geom" type="cylinder" fromto="0 0 0 0 0.08 0" size="0.025" mass="0.696"/>
        <body name="FL_thigh" pos="0 0.08 0">
          <joint name="FL_thigh_joint" axis="0 1 0" range="-0.686 4.501"/>
          <geom name="FL_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.213" size="0.02" mass="1.013"/>
          <body name="FL_calf" pos="0 0 -0.213">
            <joint name="FL_calf_joint" axis="0 1 0" range="-2.818 -0.888"/>
            <geom name="FL_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.213" size="0.015" mass="0.166"/>
            <geom name="FL_foot" type="sphere" pos="0 0 -0.213" size="0.02" mass="0.06" condim="3" conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>
      <body name="RR_hip" pos="-0.183 -0.047 0">
        <joint name="RR_hip_joint" axis="1 0 0" range="-0.863 0.863"/>
        <geom name="RR_hip_geom" type="cylinder" fromto="0 0 0 0 -0.08 0" size="0.025" mass="0.696"/>
        <body name="RR_thigh" pos="0 -0.08 0">
          <joint name="RR_thigh_joint" axis="0 1 0" range="-0.686 4.501"/>
          <geom name="RR_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.213" size="0.02" mass="1.013"/>
          <body name="RR_calf" pos="0 0 -0.213">
            <joint name="RR_calf_joint" axis="0 1 0" range="-2.818 -0.888"/>
            <geom name="RR_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.213" size="0.015" mass="0.166"/>
            <geom name="RR_foot" type="sphere" pos="0 0 -0.213" size="0.02" mass="0.06" condim="3" conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>
      <body name="RL_hip" pos="-0.183 0.047 0">
        <joint name="RL_hip_joint" axis="1 0 0" range="-0.863 0.863"/>
        <geom name="RL_hip_geom" type="cylinder" fromto="0 0 0 0 0.08 0" size="0.025" mass="0.696"/>
        <body name="RL_thigh" pos="0 0.08 0">
          <joint name="RL_thigh_joint" axis="0 1 0" range="-0.686 4.501"/>
          <geom name="RL_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.213" size="0.02" mass="1.013"/>
          <body name="RL_calf" pos="0 0 -0.213">
            <joint name="RL_calf_joint" axis="0 1 0" range="-2.818 -0.888"/>
            <geom name="RL_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.213" size="0.015" mass="0.166"/>
            <geom name="RL_foot" type="sphere" pos="0 0 -0.213" size="0.02" mass="0.06" condim="3" conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="FR_hip_motor" joint="FR_hip_joint" ctrlrange="-0.863 0.863" kp="{kp}"/>
    <position name="FR_thigh_motor" joint="FR_thigh_joint" ctrlrange="-0.686 4.501" kp="{kp}"/>
    <position name="FR_calf_motor" joint="FR_calf_joint" ctrlrange="-2.818 -0.888" kp="{kp}"/>
    <position name="FL_hip_motor" joint="FL_hip_joint" ctrlrange="-0.863 0.863" kp="{kp}"/>
    <position name="FL_thigh_motor" joint="FL_thigh_joint" ctrlrange="-0.686 4.501" kp="{kp}"/>
    <position name="FL_calf_motor" joint="FL_calf_joint" ctrlrange="-2.818 -0.888" kp="{kp}"/>
    <position name="RR_hip_motor" joint="RR_hip_joint" ctrlrange="-0.863 0.863" kp="{kp}"/>
    <position name="RR_thigh_motor" joint="RR_thigh_joint" ctrlrange="-0.686 4.501" kp="{kp}"/>
    <position name="RR_calf_motor" joint="RR_calf_joint" ctrlrange="-2.818 -0.888" kp="{kp}"/>
    <position name="RL_hip_motor" joint="RL_hip_joint" ctrlrange="-0.863 0.863" kp="{kp}"/>
    <position name="RL_thigh_motor" joint="RL_thigh_joint" ctrlrange="-0.686 4.501" kp="{kp}"/>
    <position name="RL_calf_motor" joint="RL_calf_joint" ctrlrange="-2.818 -0.888" kp="{kp}"/>
  </actuator>
</mujoco>
"""


def build_go1_xml(
    terrain_type: TerrainType = TerrainType.FLAT,
    terrain_params: dict | None = None,
    sim_params: SimParams | None = None,
) -> str:
    """Build a Go1 MJCF string with the requested terrain and sim params."""
    sim_params = sim_params or SimParams.default()
    terrain_params = dict(terrain_params or {})
    terrain_params.setdefault("friction", sim_params.global_friction)
    terrain_xml = generate_terrain_xml(terrain_type, params=terrain_params)
    return GO1_MODEL_TEMPLATE.format(
        terrain_xml=terrain_xml,
        damping=2.0 * sim_params.joint_damping_scale * sim_params.kd_scale,
        armature=0.02 * sim_params.joint_armature_scale,
        friction=sim_params.global_friction,
        kp=40.0 * sim_params.kp_scale,
    )


class Go1SimEnv(gym.Env if gym is not None else object):
    """A lightweight MuJoCo quadruped env tailored to the local Go1 demo."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        terrain_type: TerrainType = TerrainType.FLAT,
        terrain_params: dict | None = None,
        sim_params: SimParams | None = None,
        max_steps: int = 1000,
        frame_skip: int = 10,
        render_mode: str | None = None,
    ):
        if gym is None or spaces is None or mujoco is None:  # pragma: no cover - optional dependency
            raise ImportError("Go1SimEnv requires gymnasium and mujoco; install with `.[sim]`.")

        self.terrain_type = terrain_type
        self.terrain_params = terrain_params or {}
        self.sim_params = sim_params or SimParams.default()
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(
            build_go1_xml(
                terrain_type=terrain_type,
                terrain_params=self.terrain_params,
                sim_params=self.sim_params,
            )
        )
        self.data = mujoco.MjData(self.model)
        self.control_dt = self.model.opt.timestep * self.frame_skip
        self._step_count = 0
        self._rng = np.random.default_rng(0)
        self._renderer = None
        self._fall_grace_steps = 0 if terrain_type == TerrainType.FLAT else 15
        self._target_clearance = GO1_TARGET_CLEARANCE
        self._reference_upright = 1.0

        if self.sim_params.mass_scale != 1.0:
            self.model.body_mass[1:] *= self.sim_params.mass_scale

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(35,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )
        self._action_history: deque[np.ndarray] = deque()
        self._last_ctrl = GO1_STANDING_POSE.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._set_standing_pose()
        self._step_count = 0
        self._action_history.clear()
        self._last_ctrl = GO1_STANDING_POSE.copy()
        observation = self._get_observation()
        return observation, {}

    def settle_joint_positions(
        self,
        joint_positions: np.ndarray,
        *,
        settle_steps: int = 200,
    ) -> np.ndarray:
        """Re-settle the robot around a baseline joint pose and refresh terrain references."""
        clipped = np.clip(
            np.asarray(joint_positions, dtype=np.float32),
            GO1_JOINT_LIMITS[:, 0],
            GO1_JOINT_LIMITS[:, 1],
        )
        self.data.qvel[:] = 0.0
        self.data.qpos[7:19] = clipped
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = clipped
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        self._action_history.clear()
        self._last_ctrl = clipped.copy()
        settled_clearance = float(self._terrain_clearance())
        self._target_clearance = (
            max(settled_clearance, GO1_TARGET_CLEARANCE)
            if self.terrain_type == TerrainType.FLAT
            else max(settled_clearance, 0.12)
        )
        self._reference_upright = float(max(self._upright_alignment(), 0.0))
        return self._get_observation()

    def step(self, action: np.ndarray):
        target_positions = self.denormalize_action(action)
        return self.step_target_positions(target_positions)

    def step_target_positions(self, target_positions: np.ndarray):
        target_positions = np.clip(
            np.asarray(target_positions, dtype=np.float32),
            GO1_JOINT_LIMITS[:, 0],
            GO1_JOINT_LIMITS[:, 1],
        )
        delayed_action = self._apply_latency(target_positions)
        self.data.ctrl[:] = delayed_action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        observation = self._get_observation()
        terminated = self._is_fallen()
        reward = self._compute_reward(delayed_action, terminated=terminated)
        truncated = self._step_count >= self.max_steps
        info = {
            "x_velocity": float(self.data.qvel[0]),
            "height": float(self.data.qpos[2]),
            "terrain": self.terrain_type.value,
        }
        self._last_ctrl = delayed_action.copy()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":  # pragma: no cover - render path
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:  # pragma: no cover - render path
            self._renderer.close()
            self._renderer = None

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized actions into a safe band around the standing pose."""
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return GO1_STANDING_POSE + clipped * GO1_POLICY_ACTION_RANGE

    def normalize_action(self, target_positions: np.ndarray) -> np.ndarray:
        """Map absolute target positions into the normalized policy space."""
        return np.clip(
            (np.asarray(target_positions) - GO1_STANDING_POSE) / GO1_POLICY_ACTION_RANGE,
            -1.0,
            1.0,
        )

    def _set_standing_pose(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        base_position, base_quat = self._terrain_reset_pose()
        self.data.qpos[0:3] = base_position
        self.data.qpos[3:7] = base_quat
        self.data.qpos[7:19] = GO1_STANDING_POSE
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = GO1_STANDING_POSE
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        settled_clearance = float(self._terrain_clearance())
        self._target_clearance = (
            max(settled_clearance, GO1_TARGET_CLEARANCE)
            if self.terrain_type == TerrainType.FLAT
            else max(settled_clearance, 0.12)
        )
        self._reference_upright = float(max(self._upright_alignment(), 0.0))

    def _terrain_reset_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Pick a stable spawn pose for the current terrain."""
        base_position = np.array([0.0, 0.0, 0.35], dtype=np.float32)
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if self.terrain_type == TerrainType.SLOPE:
            base_position = np.array([-0.5, 0.0, 0.42], dtype=np.float32)
        elif self.terrain_type == TerrainType.STAIRS:
            base_position = np.array([-0.55, 0.0, 0.42], dtype=np.float32)

        return base_position, base_quat

    def _apply_latency(self, target_positions: np.ndarray) -> np.ndarray:
        delay_steps = int(round((self.sim_params.actuator_latency_ms / 1000.0) / self.control_dt))
        self._action_history.append(target_positions.copy())
        while len(self._action_history) > max(delay_steps + 1, 1):
            self._action_history.popleft()
        if delay_steps <= 0 or len(self._action_history) <= delay_steps:
            return target_positions
        return self._action_history[0]

    def _get_observation(self) -> np.ndarray:
        quat = self.data.qpos[3:7]
        observation = np.concatenate(
            [
                self.data.qpos[2:7],
                self.data.qpos[7:19],
                self.data.qvel[0:6],
                self.data.qvel[6:18],
            ]
        ).astype(np.float32)
        if self.sim_params.observation_noise_std > 0:
            observation = observation + self._rng.normal(
                0.0,
                self.sim_params.observation_noise_std,
                size=observation.shape,
            ).astype(np.float32)
            observation[1:5] = quat
        return observation

    def _is_fallen(self) -> bool:
        if self._step_count < self._fall_grace_steps:
            return False
        clearance = self._terrain_clearance()
        upright_alignment = self._upright_alignment()
        clearance_threshold = (
            max(0.12, 0.45 * self._target_clearance)
            if self.terrain_type == TerrainType.FLAT
            else max(0.08, 0.7 * self._target_clearance)
        )
        upright_threshold = (
            0.55
            if self.terrain_type == TerrainType.FLAT
            else max(0.05, self._reference_upright - 0.35)
        )
        return bool(clearance < clearance_threshold or upright_alignment < upright_threshold)

    def _compute_reward(self, ctrl: np.ndarray, *, terminated: bool) -> float:
        forward_velocity = self._forward_speed()
        clearance = self._terrain_clearance()
        upright_alignment = self._upright_alignment()
        joint_positions = self.data.qpos[7:19]
        joint_acc = np.clip(self.data.qacc[6:], -30.0, 30.0)

        alive_bonus = 1.0
        forward_reward = 2.0 * float(np.clip(forward_velocity, -0.5, 1.5))
        height_penalty = -4.0 * abs(clearance - self._target_clearance)
        orientation_penalty = -3.0 * max(self._reference_upright - upright_alignment, 0.0)
        posture_penalty = -0.05 * float(np.sum(np.square(joint_positions - GO1_STANDING_POSE)))
        action_rate_penalty = -0.02 * float(np.sum(np.square(ctrl - self._last_ctrl)))
        smoothness_penalty = -0.0002 * float(np.sum(np.square(joint_acc)))
        fall_penalty = -50.0 if terminated else 0.0

        return (
            alive_bonus
            + forward_reward
            + height_penalty
            + orientation_penalty
            + posture_penalty
            + action_rate_penalty
            + smoothness_penalty
            + fall_penalty
        )

    def _terrain_normal(self, x_position: float | None = None) -> np.ndarray:
        """Return the surface normal for the current terrain near the robot base."""
        x_position = float(self.data.qpos[0]) if x_position is None else x_position
        if self.terrain_type == TerrainType.SLOPE:
            approach_length = float(self.terrain_params.get("approach_length", 1.0))
            if x_position < 0.0 or x_position < approach_length:
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)
            angle_deg = float(self.terrain_params.get("angle_deg", 15.0))
            angle_rad = np.deg2rad(angle_deg)
            normal = np.array([-np.sin(angle_rad), 0.0, np.cos(angle_rad)], dtype=np.float32)
            return normal / np.linalg.norm(normal)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _terrain_height(self, x_position: float) -> float:
        """Approximate terrain surface height in world-z at the robot base x position."""
        if self.terrain_type == TerrainType.SLOPE:
            if x_position <= 0.0:
                return 0.0
            angle_deg = float(self.terrain_params.get("angle_deg", 15.0))
            angle_rad = np.deg2rad(angle_deg)
            ramp_length = float(self.terrain_params.get("ramp_length", 3.0))
            ramp_progress = min(max(x_position, 0.0), ramp_length)
            return float(np.tan(angle_rad) * ramp_progress)
        if self.terrain_type == TerrainType.STAIRS:
            step_height = float(self.terrain_params.get("step_height", 0.04))
            step_width = float(self.terrain_params.get("step_width", 0.4))
            n_steps = int(self.terrain_params.get("n_steps", 8))
            if x_position < 0.0:
                return 0.0
            step_index = min(int(x_position // step_width), n_steps - 1)
            return float(max(step_index, 0) * step_height)
        return 0.0

    def _terrain_clearance(self) -> float:
        """Measure body height relative to the local terrain."""
        base_position = self.data.qpos[0:3]
        if self.terrain_type == TerrainType.STAIRS:
            return float(base_position[2] - self._terrain_height(float(base_position[0])))
        x_position = float(base_position[0])
        normal = self._terrain_normal(x_position)
        terrain_point = np.array([x_position, 0.0, self._terrain_height(x_position)], dtype=np.float32)
        return float(np.dot(base_position - terrain_point, normal))

    def _forward_axis(self) -> np.ndarray:
        """Return the local tangent direction used for velocity tracking rewards."""
        if self.terrain_type == TerrainType.SLOPE:
            if float(self.data.qpos[0]) <= 0.0:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            angle_deg = float(self.terrain_params.get("angle_deg", 15.0))
            angle_rad = np.deg2rad(angle_deg)
            axis = np.array([np.cos(angle_rad), 0.0, np.sin(angle_rad)], dtype=np.float32)
            return axis / np.linalg.norm(axis)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def _forward_speed(self) -> float:
        """Project base linear velocity onto the terrain tangent."""
        base_linear_velocity = np.asarray(self.data.qvel[0:3], dtype=np.float32)
        return float(np.dot(base_linear_velocity, self._forward_axis()))

    def _upright_alignment(self) -> float:
        """Compare the body up-vector against the terrain normal."""
        quat_wxyz = np.asarray(self.data.qpos[3:7], dtype=np.float32)
        body_up = _rotate_vector_by_quaternion(
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            quat_wxyz,
        )
        return float(np.clip(np.dot(body_up, self._terrain_normal(float(self.data.qpos[0]))), -1.0, 1.0))


class TrottingGaitController:
    """Simple trotting gait used by the local demo and tests."""

    def __init__(
        self,
        frequency: float = 2.0,
        amplitude_hip: float = 0.15,
        amplitude_thigh: float = 0.4,
        amplitude_calf: float = 0.6,
        standing_thigh: float = 0.8,
        standing_calf: float = -1.6,
    ):
        self.frequency = frequency
        self.amplitude_hip = amplitude_hip
        self.amplitude_thigh = amplitude_thigh
        self.amplitude_calf = amplitude_calf
        self.standing_thigh = standing_thigh
        self.standing_calf = standing_calf
        self.phases = {"FR": 0.0, "FL": np.pi, "RR": np.pi, "RL": 0.0}

    def get_action(self, t: float) -> np.ndarray:
        action = np.zeros(12, dtype=np.float32)
        for idx, leg in enumerate(LEG_NAMES):
            phase = 2 * np.pi * self.frequency * t + self.phases[leg]
            action[idx * 3 + 0] = self.amplitude_hip * np.sin(phase * 0.5)
            action[idx * 3 + 1] = self.standing_thigh + self.amplitude_thigh * np.sin(phase)
            action[idx * 3 + 2] = self.standing_calf + self.amplitude_calf * np.sin(phase + 0.3)
        return action


class BoundingGaitController:
    """Bounding gait controller used by the demo."""

    def __init__(
        self,
        frequency: float = 2.5,
        amplitude_thigh: float = 0.5,
        amplitude_calf: float = 0.7,
        standing_thigh: float = 0.8,
        standing_calf: float = -1.6,
    ):
        self.frequency = frequency
        self.amplitude_thigh = amplitude_thigh
        self.amplitude_calf = amplitude_calf
        self.standing_thigh = standing_thigh
        self.standing_calf = standing_calf
        self.phases = {"FR": 0.0, "FL": 0.0, "RR": np.pi, "RL": np.pi}

    def get_action(self, t: float) -> np.ndarray:
        action = np.zeros(12, dtype=np.float32)
        for idx, leg in enumerate(LEG_NAMES):
            phase = 2 * np.pi * self.frequency * t + self.phases[leg]
            action[idx * 3 + 1] = self.standing_thigh + self.amplitude_thigh * np.sin(phase)
            action[idx * 3 + 2] = self.standing_calf + self.amplitude_calf * np.sin(phase + 0.3)
        return action


def replay_go1_episode(
    action_sequence: np.ndarray,
    sim_params: SimParams | None = None,
    terrain_type: TerrainType = TerrainType.FLAT,
    terrain_params: dict | None = None,
    robot_spec: RobotSpec | None = None,
) -> Episode:
    """Replay a sequence of absolute joint targets and return an OLSD Episode."""
    env = Go1SimEnv(
        terrain_type=terrain_type,
        terrain_params=terrain_params,
        sim_params=sim_params,
        max_steps=len(action_sequence),
    )
    robot = robot_spec or load_robot_by_id("go1")

    env.reset()
    steps: list[Step] = []
    for idx, target_positions in enumerate(action_sequence):
        next_observation, reward, terminated, truncated, _ = env.step_target_positions(target_positions)
        steps.append(
            Step(
                observation=_observation_from_array(next_observation, env),
                action=Action(
                    values=np.asarray(target_positions, dtype=np.float32).tolist(),
                    control_mode=ControlMode.POSITION,
                ),
                reward=float(reward),
                done=bool(terminated),
                truncated=bool(truncated),
                timestamp=idx * env.control_dt,
            )
        )
        if terminated or truncated:
            break

    env.close()
    metadata = EpisodeMetadata(
        robot=robot,
        terrain=TerrainSpec(terrain_type=terrain_type),
        source=DataSource.SIMULATION,
        simulator="mujoco",
        sampling_rate_hz=1.0 / env.control_dt,
        success=not any(step.done for step in steps),
    )
    return Episode(
        episode_id=f"go1_replay_{terrain_type.value}",
        steps=steps,
        metadata=metadata,
    )


def rollout_controller(
    controller,
    n_steps: int = 500,
    terrain_type: TerrainType = TerrainType.FLAT,
    terrain_params: dict | None = None,
    sim_params: SimParams | None = None,
    robot_spec: RobotSpec | None = None,
) -> Episode:
    """Roll out a gait controller through the Go1 env."""
    action_sequence = np.stack(
        [controller.get_action(idx * 0.02) for idx in range(n_steps)],
        axis=0,
    )
    return replay_go1_episode(
        action_sequence=action_sequence,
        sim_params=sim_params,
        terrain_type=terrain_type,
        terrain_params=terrain_params,
        robot_spec=robot_spec,
    )


def _observation_from_array(observation: np.ndarray, env: Go1SimEnv) -> Observation:
    """Convert the flat env observation into an OLSD Observation model."""
    return Observation(
        joint_positions=observation[5:17].tolist(),
        joint_velocities=observation[23:35].tolist(),
        imu_orientation=observation[1:5].tolist(),
        base_position=[float(env.data.qpos[0]), float(env.data.qpos[1]), float(env.data.qpos[2])],
        base_velocity=observation[17:20].tolist(),
        base_angular_velocity=observation[20:23].tolist(),
    )


def _rotate_vector_by_quaternion(vector: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a MuJoCo-style quaternion [w, x, y, z]."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float32)
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation @ np.asarray(vector, dtype=np.float32)
