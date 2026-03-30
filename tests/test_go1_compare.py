"""Tests for the strict Go1 head-to-head comparison scaffold."""

from __future__ import annotations

import numpy as np

from olsd.sim2real.go1_compare import (
    DEFAULT_SELECTED_BASELINE,
    SHARED_GO1_METRICS,
    _build_wtw_command_vector,
    _rotate_world_vector_into_body_frame,
    build_go1_head_to_head_report,
    summarize_episode_records,
)


class TestGo1Compare:
    def test_summarize_episode_records(self):
        summary = summarize_episode_records(
            [
                {
                    "success": True,
                    "fall": False,
                    "episode_length": 1000,
                    "forward_velocity_mean": 0.95,
                },
                {
                    "success": False,
                    "fall": True,
                    "episode_length": 120,
                    "forward_velocity_mean": 0.20,
                },
            ]
        )

        assert summary["episode_count"] == 2
        assert summary["shared_metrics"] == SHARED_GO1_METRICS
        assert summary["success_rate"] == 0.5
        assert summary["fall_count"] == 1
        assert summary["fall_rate"] == 0.5
        assert summary["episode_length_mean"] == 560.0
        assert summary["forward_velocity_mean"] == 0.575

    def test_build_head_to_head_report(self):
        report = build_go1_head_to_head_report(
            baselines={
                "olsd_v2_canonical": {
                    "label": "OLSD v2 canonical Go1 policies",
                    "terrains": {
                        "flat": {"success_rate": 1.0},
                        "slope": {"success_rate": 1.0},
                        "stairs": {"success_rate": 1.0},
                    },
                }
            },
            selected_external_baseline=dict(DEFAULT_SELECTED_BASELINE),
            n_eval_episodes=20,
            horizon=1000,
            seed=0,
        )

        assert report["schema_version"] == "1.0"
        assert report["robot_id"] == "unitree_go1"
        assert report["protocol"]["terrains"] == ["flat", "slope", "stairs"]
        assert report["protocol"]["shared_metrics"] == SHARED_GO1_METRICS
        assert report["selected_external_baseline"]["baseline_id"] == "walk_these_ways_pretrain_v0"
        assert "olsd_v2_canonical" in report["baselines"]

    def test_build_wtw_command_vector(self):
        command_vector = _build_wtw_command_vector(
            {
                "x_vel_cmd": 0.6,
                "y_vel_cmd": 0.1,
                "yaw_vel_cmd": -0.2,
                "body_height_cmd": 0.0,
                "step_frequency_cmd": 3.0,
                "gait_phase_cmd": 0.5,
                "gait_offset_cmd": 0.1,
                "gait_bound_cmd": 0.2,
                "gait_duration_cmd": 0.5,
                "footswing_height_cmd": 0.08,
                "pitch_cmd": 0.0,
                "roll_cmd": 0.0,
                "stance_width_cmd": 0.25,
                "stance_length_cmd": 0.4,
                "aux_reward_cmd": 0.0,
            }
        )

        assert command_vector.shape == (15,)
        assert np.allclose(command_vector[:5], [0.6, 0.1, -0.2, 0.0, 3.0])
        assert command_vector[-1] == 0.0

    def test_rotate_world_vector_into_body_frame_identity(self):
        rotated = _rotate_world_vector_into_body_frame(
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

        assert np.allclose(rotated, [0.0, 0.0, -1.0])
