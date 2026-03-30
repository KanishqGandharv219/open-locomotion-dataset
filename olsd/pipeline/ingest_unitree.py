"""
Unitree G1/H1 Retargeted Motion Ingestor — Benchmark-Only.

Source: https://huggingface.co/datasets/openhe/g1-retargeted-motions
Format: Python pickle (.pkl) files with retargeted motion capture data.

⚠️ LICENSE: Must be verified on HuggingFace before redistribution.
   This ingestor is gated for BENCHMARK-ONLY use. Data ingested here
   should NOT be redistributed as part of the OLSD public release unless
   the license is confirmed as permissive (CC-BY, MIT, Apache, etc.).

Fields in pickle:
    root_trans_offset  -> base position
    root_rot           -> base orientation (quaternion)
    dof (23 values)    -> joint positions
    pose_aa            -> axis-angle pose (stored in step.info)
    contact_mask       -> foot contact binary flags
    fps                -> typically 30
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from uuid import uuid4

import numpy as np

from olsd.schema import (
    Action,
    ControlMode,
    DataSource,
    Episode,
    EpisodeMetadata,
    Morphology,
    Observation,
    RobotSpec,
    Step,
    TerrainSpec,
    TerrainType,
)

logger = logging.getLogger(__name__)


# Unitree G1: 23 DOF humanoid
# Torso: 3 (waist pitch, roll, yaw)
# Arms: 8 per arm × 2 = 16 (shoulder×3, elbow×2, wrist×3)
# Legs: ... wait, G1 is 23 DOF, various configs exist
# Standard: 3 torso + 7×2 arms + 3×2 legs = 23

UNITREE_ROBOTS = {
    "g1": RobotSpec(
        robot_id="unitree_g1",
        robot_name="Unitree G1",
        morphology=Morphology.HUMANOID,
        n_joints=23,
        n_actuators=23,
        mass_kg=35.0,
        standing_height_m=1.27,
        manufacturer="Unitree Robotics",
        description="23-DOF humanoid (3 torso + 7×2 arms + 3×2 legs). "
                    "Data from kinematic retargeting — not dynamically feasible.",
    ),
    "h1": RobotSpec(
        robot_id="unitree_h1",
        robot_name="Unitree H1",
        morphology=Morphology.HUMANOID,
        n_joints=19,
        n_actuators=19,
        mass_kg=47.0,
        standing_height_m=1.80,
        manufacturer="Unitree Robotics",
        description="19-DOF humanoid.",
    ),
    "h1_2": RobotSpec(
        robot_id="unitree_h1_2",
        robot_name="Unitree H1-2",
        morphology=Morphology.HUMANOID,
        n_joints=19,
        n_actuators=19,
        mass_kg=47.0,
        standing_height_m=1.80,
        manufacturer="Unitree Robotics",
        description="19-DOF humanoid (v2).",
    ),
}


def from_unitree_retargeted(
    pkl_path: str | Path,
    robot_type: str = "g1",
    max_episodes: int | None = None,
) -> list[Episode]:
    """
    Convert Unitree retargeted motion .pkl file(s) to OLSD Episodes.

    Field mapping:
        dof              (N values)   -> observation.joint_positions
        finite_diff(dof) (computed)   -> observation.joint_velocities
        root_trans_offset             -> observation.base_position
        root_rot                      -> observation.imu_orientation
        contact_mask                  -> observation.contact_binary
        pose_aa                       -> step.info["pose_aa"]

    ⚠️ Velocities are derived via finite differences at source fps (typically 30).
       Torques are NOT available (kinematic-only retargeting).

    Args:
        pkl_path: Path to .pkl file or directory of .pkl files.
        robot_type: "g1", "h1", or "h1_2".
        max_episodes: Maximum episodes to load.

    Returns:
        List of OLSD Episodes (benchmark-only, check license before redistribution).
    """
    path = Path(pkl_path)
    robot = UNITREE_ROBOTS.get(robot_type)
    if robot is None:
        raise ValueError(f"Unknown robot_type '{robot_type}'. Options: {list(UNITREE_ROBOTS.keys())}")

    episodes: list[Episode] = []

    if path.is_file() and path.suffix == ".pkl":
        pkl_files = [path]
    elif path.is_dir():
        pkl_files = sorted(path.glob("*.pkl"))
    else:
        raise FileNotFoundError(f"No .pkl files found at {path}")

    for pkl_file in pkl_files:
        if max_episodes and len(episodes) >= max_episodes:
            break

        try:
            ep = _pkl_to_episode(pkl_file, robot)
            if ep is not None:
                episodes.append(ep)
        except Exception as e:
            logger.warning(f"Failed to load {pkl_file}: {e}")

    logger.info(f"Unitree {robot_type}: loaded {len(episodes)} episodes, "
                f"{sum(e.n_steps for e in episodes)} total steps")
    return episodes


def _pkl_to_episode(pkl_path: Path, robot: RobotSpec) -> Episode | None:
    """Convert a single .pkl motion file to an Episode."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Handle both dict and list-of-dicts formats
    if isinstance(data, list):
        # Multiple motions in one file
        motions = data
    elif isinstance(data, dict):
        motions = [data]
    else:
        logger.warning(f"Unexpected data type in {pkl_path}: {type(data)}")
        return None

    # Process first motion (or concatenate if needed)
    motion = motions[0]

    # Extract DOF (joint positions)
    dof = _get_field(motion, ["dof", "dof_pos", "joint_positions"])
    if dof is None:
        logger.warning(f"No DOF data in {pkl_path}")
        return None

    dof = np.array(dof, dtype=np.float32)
    if dof.ndim == 1:
        # Single frame — skip
        return None

    n_frames, n_dof = dof.shape
    fps = float(motion.get("fps", 30))
    dt = 1.0 / fps

    # Compute joint velocities via finite differences
    jv = np.zeros_like(dof)
    jv[1:] = (dof[1:] - dof[:-1]) * fps

    # Optional fields
    root_trans = _get_field(motion, ["root_trans_offset", "root_trans", "trans"])
    root_rot = _get_field(motion, ["root_rot", "root_rotation", "rotation"])
    contact_mask = _get_field(motion, ["contact_mask", "contacts", "foot_contacts"])
    pose_aa = _get_field(motion, ["pose_aa", "pose", "axis_angle"])

    if root_trans is not None:
        root_trans = np.array(root_trans, dtype=np.float32)
    if root_rot is not None:
        root_rot = np.array(root_rot, dtype=np.float32)
    if contact_mask is not None:
        contact_mask = np.array(contact_mask, dtype=bool)
    if pose_aa is not None:
        pose_aa = np.array(pose_aa, dtype=np.float32)

    # Adjust robot spec if DOF doesn't match expected
    if n_dof != robot.n_joints:
        robot = RobotSpec(
            robot_id=robot.robot_id,
            robot_name=robot.robot_name,
            morphology=robot.morphology,
            n_joints=n_dof,
            n_actuators=n_dof,
            mass_kg=robot.mass_kg,
            standing_height_m=robot.standing_height_m,
            manufacturer=robot.manufacturer,
            description=f"{robot.description} (actual DOF: {n_dof})",
        )

    steps = []
    for i in range(n_frames):
        obs = Observation(
            joint_positions=dof[i].tolist(),
            joint_velocities=jv[i].tolist(),
        )

        # Optional: base position
        if root_trans is not None and i < len(root_trans):
            pos = root_trans[i]
            obs.base_position = pos[:3].tolist() if len(pos) >= 3 else pos.tolist()

        # Optional: orientation (convert to quaternion if needed)
        if root_rot is not None and i < len(root_rot):
            rot = root_rot[i]
            if len(rot) == 4:
                obs.imu_orientation = rot.tolist()
            elif len(rot) == 3:
                # Axis-angle -> quaternion (simplified)
                angle = np.linalg.norm(rot)
                if angle > 1e-8:
                    axis = rot / angle
                    w = np.cos(angle / 2)
                    xyz = axis * np.sin(angle / 2)
                    obs.imu_orientation = [float(w), float(xyz[0]), float(xyz[1]), float(xyz[2])]

        # Optional: contact
        if contact_mask is not None and i < len(contact_mask):
            obs.contact_binary = contact_mask[i].tolist()

        # Action: use joint positions as target (kinematic data)
        action = Action(values=dof[i].tolist(), control_mode=ControlMode.POSITION)

        # Store pose_aa in info dict
        info = None
        if pose_aa is not None and i < len(pose_aa):
            info = {"pose_aa": pose_aa[i].tolist()}

        step = Step(
            observation=obs,
            action=action,
            reward=None,
            done=(i == n_frames - 1),
            timestamp=i * dt,
            info=info,
        )
        steps.append(step)

    if not steps:
        return None

    metadata = EpisodeMetadata(
        robot=robot,
        terrain=TerrainSpec(terrain_type=TerrainType.FLAT),
        source=DataSource.MIXED,  # kinematic retargeting from human mocap
        sampling_rate_hz=fps,
        external_dataset="unitree_retargeted",
        external_episode_id=pkl_path.stem,
        task_description=f"Retargeted human motion ({pkl_path.stem})",
        license="check-before-redistribution",  # ⚠️ must verify
    )

    return Episode(
        episode_id=str(uuid4()),
        steps=steps,
        metadata=metadata,
    )


def _get_field(data: dict, keys: list[str]):
    """Try multiple key names to extract a field from a dict."""
    for key in keys:
        if key in data:
            return data[key]
    return None
