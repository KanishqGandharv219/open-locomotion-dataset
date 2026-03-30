"""
Cross-Embodiment Dimension Alignment — Training-Time Only.

These utilities align observation/action spaces across different robot morphologies
for multi-embodiment learning (e.g., Multi-Loco style training).

IMPORTANT: Padding and masking are computed ON-THE-FLY by the DataLoader.
           Raw joint_positions / joint_velocities remain the canonical stored format.
           Global stats (max_dof, normalization) are stored once in meta/normalization.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_max_dof(episodes: list[Any]) -> int:
    """Find the maximum DOF (joint count) across all episodes.

    Args:
        episodes: List of Episode objects.

    Returns:
        Maximum number of joints found across all episodes.
    """
    max_dof = 0
    for ep in episodes:
        n = len(ep.steps[0].observation.joint_positions)
        max_dof = max(max_dof, n)
    return max_dof


def create_active_mask(n_joints: int, max_dof: int) -> np.ndarray:
    """Create a boolean mask: True for real joint dims, False for padding.

    Args:
        n_joints: Number of actual joints for this robot.
        max_dof: Maximum DOF across all robots in the dataset.

    Returns:
        Boolean array of shape [max_dof].
    """
    mask = np.zeros(max_dof, dtype=bool)
    mask[:n_joints] = True
    return mask


def pad_array(arr: np.ndarray, max_dof: int) -> np.ndarray:
    """Zero-pad a 1D joint array to max_dof length.

    Args:
        arr: Array of shape [n_joints].
        max_dof: Target padded length.

    Returns:
        Zero-padded array of shape [max_dof].
    """
    if len(arr) >= max_dof:
        return arr[:max_dof]
    padded = np.zeros(max_dof, dtype=arr.dtype)
    padded[: len(arr)] = arr
    return padded


def pad_batch(
    data: np.ndarray,
    max_dof: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad a batch of joint arrays and return (padded, mask).

    Args:
        data: Array of shape [batch, n_joints].
        max_dof: Target padded dimension.

    Returns:
        Tuple of (padded_data [batch, max_dof], active_mask [max_dof]).
    """
    batch_size, n_joints = data.shape
    if n_joints >= max_dof:
        return data[:, :max_dof], np.ones(max_dof, dtype=bool)

    padded = np.zeros((batch_size, max_dof), dtype=data.dtype)
    padded[:, :n_joints] = data
    mask = create_active_mask(n_joints, max_dof)
    return padded, mask


def compute_normalization_stats(episodes: list[Any]) -> dict:
    """Compute per-robot MinMax normalization statistics.

    Stats are computed per robot_id so that normalization respects
    different physical scales across embodiments.

    Args:
        episodes: List of Episode objects.

    Returns:
        Dict keyed by robot_id with min/max/mean/std per feature.
        Saved once to meta/normalization.json.
    """
    from collections import defaultdict

    by_robot: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"jp": [], "jv": [], "act": []}
    )

    for ep in episodes:
        rid = ep.metadata.robot.robot_id
        data = ep.to_numpy()
        by_robot[rid]["jp"].append(data["joint_positions"])
        by_robot[rid]["jv"].append(data["joint_velocities"])
        by_robot[rid]["act"].append(data["actions"])

    stats: dict = {}
    for rid, arrays in by_robot.items():
        robot_stats: dict = {}
        for key in ["jp", "jv", "act"]:
            if not arrays[key]:
                continue
            concat = np.concatenate(arrays[key])
            robot_stats[key] = {
                "min": concat.min(axis=0).tolist(),
                "max": concat.max(axis=0).tolist(),
                "mean": concat.mean(axis=0).tolist(),
                "std": concat.std(axis=0).tolist(),
                "n_dims": int(concat.shape[1]) if concat.ndim > 1 else 1,
            }
        stats[rid] = robot_stats

    # Global max_dof
    all_dims = []
    for rs in stats.values():
        if "jp" in rs:
            all_dims.append(rs["jp"]["n_dims"])
    stats["_global"] = {"max_dof": max(all_dims) if all_dims else 0}

    return stats


def save_normalization_stats(stats: dict, output_dir: str | Path) -> Path:
    """Save normalization stats to meta/normalization.json.

    Args:
        stats: Stats dict from compute_normalization_stats().
        output_dir: Dataset output directory.

    Returns:
        Path to the saved JSON file.
    """
    out = Path(output_dir) / "meta" / "normalization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved normalization stats to {out}")
    return out


def load_normalization_stats(dataset_dir: str | Path) -> dict:
    """Load normalization stats from meta/normalization.json.

    Args:
        dataset_dir: Dataset directory.

    Returns:
        Stats dict.
    """
    path = Path(dataset_dir) / "meta" / "normalization.json"
    with open(path) as f:
        return json.load(f)


def normalize_array(
    arr: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """MinMax normalize an array to [0, 1] range.

    Args:
        arr: Input array.
        min_vals: Per-dimension minimums (from stats).
        max_vals: Per-dimension maximums (from stats).
        eps: Small constant to prevent division by zero.

    Returns:
        Normalized array in [0, 1].
    """
    return (arr - min_vals) / (max_vals - min_vals + eps)


def denormalize_array(
    arr: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Reverse MinMax normalization.

    Args:
        arr: Normalized array in [0, 1].
        min_vals: Per-dimension minimums.
        max_vals: Per-dimension maximums.
        eps: Small constant.

    Returns:
        Denormalized array.
    """
    return arr * (max_vals - min_vals + eps) + min_vals
