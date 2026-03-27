"""
OLSD Benchmark Task Definitions.

Standardized tasks for benchmarking locomotion policies across robots and terrains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkTask:
    """Definition of a standardized benchmark task."""

    task_id: str
    name: str
    description: str
    robot_ids: list[str]
    terrain: str = "flat"
    target_speed_mps: float | None = None
    max_steps: int = 1000
    success_criteria: str = "survive"  # "survive", "reach_speed", "reach_goal"
    difficulty: str = "easy"  # "easy", "medium", "hard"

    def __str__(self) -> str:
        return f"Task({self.task_id}: {self.name})"


# ---------------------------------------------------------------------------
# Standard benchmark tasks
# ---------------------------------------------------------------------------

BENCHMARK_TASKS = {
    # Task 1: Basic forward locomotion
    "walk_flat": BenchmarkTask(
        task_id="walk_flat",
        name="Walk on Flat Ground",
        description="Walk forward on flat concrete at moderate speed without falling",
        robot_ids=["halfcheetah", "ant", "walker2d", "hopper"],
        terrain="flat",
        target_speed_mps=1.0,
        max_steps=1000,
        success_criteria="survive",
        difficulty="easy",
    ),

    # Task 2: Fast locomotion
    "run_flat": BenchmarkTask(
        task_id="run_flat",
        name="Run on Flat Ground",
        description="Run forward at high speed on flat terrain",
        robot_ids=["halfcheetah", "ant", "walker2d"],
        terrain="flat",
        target_speed_mps=3.0,
        max_steps=1000,
        success_criteria="reach_speed",
        difficulty="medium",
    ),

    # Task 3: Cross-morphology generalization
    "generalize_morphology": BenchmarkTask(
        task_id="generalize_morphology",
        name="Cross-Morphology Transfer",
        description="Train on one quadruped, evaluate on a different one",
        robot_ids=["ant"],  # train on ant, eval on others
        terrain="flat",
        max_steps=1000,
        success_criteria="survive",
        difficulty="hard",
    ),

    # Task 4: Offline RL from mixed data
    "offline_mixed": BenchmarkTask(
        task_id="offline_mixed",
        name="Offline RL from Mixed-Quality Data",
        description="Train offline on a mix of random+medium+expert data",
        robot_ids=["halfcheetah", "ant", "walker2d", "hopper"],
        terrain="flat",
        max_steps=1000,
        success_criteria="survive",
        difficulty="medium",
    ),
}


def get_task(task_id: str) -> BenchmarkTask:
    """Get a benchmark task by ID."""
    if task_id not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(BENCHMARK_TASKS.keys())}")
    return BENCHMARK_TASKS[task_id]


def list_tasks() -> list[BenchmarkTask]:
    """List all available benchmark tasks."""
    return list(BENCHMARK_TASKS.values())
