"""Strict shared-metric comparison scaffold for Go1 baselines."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from olsd.schema import TerrainType
from olsd.sim2real.go1_compare import (
    DEFAULT_SELECTED_BASELINE,
    build_go1_head_to_head_report,
    evaluate_go1_sb3_policy,
    evaluate_walk_these_ways_policy,
    resolve_go1_sim_params,
    save_go1_head_to_head_report,
)

logger = logging.getLogger("olsd.compare_go1_baselines")


def compare_go1_baselines(
    *,
    flat_policy: str | Path,
    slope_policy: str | Path,
    stairs_policy: str | Path,
    sim2real_config: str | Path | None = "configs/sim2real/go1.yaml",
    n_eval_episodes: int = 20,
    horizon: int = 1000,
    seed: int = 0,
    include_wtw: bool = False,
    wtw_root: str | Path = "external_tmp/walk_these_ways",
    external_report: str | Path | None = None,
    output_path: str | Path = "results/go1_head_to_head.json",
) -> dict:
    """Run the canonical OLSD Go1 baseline and assemble a head-to-head report."""
    sim_params = resolve_go1_sim_params(sim2real_config)
    baselines = {
        "olsd_v2_canonical": {
            "label": "OLSD v2 canonical Go1 policies",
            "policy_paths": {
                "flat": str(flat_policy),
                "slope": str(slope_policy),
                "stairs": str(stairs_policy),
            },
            "terrains": {
                "flat": evaluate_go1_sb3_policy(
                    flat_policy,
                    TerrainType.FLAT,
                    sim_params,
                    n_eval_episodes=n_eval_episodes,
                    horizon=horizon,
                    seed=seed,
                ),
                "slope": evaluate_go1_sb3_policy(
                    slope_policy,
                    TerrainType.SLOPE,
                    sim_params,
                    n_eval_episodes=n_eval_episodes,
                    horizon=horizon,
                    seed=seed + 1000,
                ),
                "stairs": evaluate_go1_sb3_policy(
                    stairs_policy,
                    TerrainType.STAIRS,
                    sim_params,
                    n_eval_episodes=n_eval_episodes,
                    horizon=horizon,
                    seed=seed + 2000,
                ),
            },
        }
    }

    selected_external_baseline = dict(DEFAULT_SELECTED_BASELINE)
    if include_wtw:
        selected_external_baseline["status"] = "integrated"
        baselines[selected_external_baseline["baseline_id"]] = {
            "label": selected_external_baseline["label"],
            "source_repo": selected_external_baseline["source_repo"],
            "checkpoint_subpath": selected_external_baseline["checkpoint_subpath"],
            "policy_root": str(Path(wtw_root)),
            "adapter": "walk_these_ways_native_jit",
            "terrains": {
                "flat": evaluate_walk_these_ways_policy(
                    repo_root=wtw_root,
                    terrain=TerrainType.FLAT,
                    sim_params=sim_params,
                    n_eval_episodes=n_eval_episodes,
                    horizon=horizon,
                    seed=seed,
                ),
                "slope": evaluate_walk_these_ways_policy(
                    repo_root=wtw_root,
                    terrain=TerrainType.SLOPE,
                    sim_params=sim_params,
                    n_eval_episodes=n_eval_episodes,
                    horizon=horizon,
                    seed=seed + 1000,
                ),
                "stairs": evaluate_walk_these_ways_policy(
                    repo_root=wtw_root,
                    terrain=TerrainType.STAIRS,
                    sim_params=sim_params,
                    n_eval_episodes=n_eval_episodes,
                    horizon=horizon,
                    seed=seed + 2000,
                ),
            },
        }
    if external_report is not None:
        with open(external_report, "r", encoding="utf-8") as handle:
            external_payload = json.load(handle)
        baseline_id = external_payload.get(
            "baseline_id",
            selected_external_baseline["baseline_id"],
        )
        selected_external_baseline["status"] = "integrated"
        baselines[baseline_id] = external_payload

    report = build_go1_head_to_head_report(
        baselines=baselines,
        selected_external_baseline=selected_external_baseline,
        n_eval_episodes=n_eval_episodes,
        horizon=horizon,
        seed=seed,
    )
    save_go1_head_to_head_report(report, output_path)
    logger.info("Saved Go1 head-to-head report to %s", output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare canonical Go1 baselines on shared physical metrics")
    parser.add_argument(
        "--flat-policy",
        default="checkpoints/phase2/go1/flat/selected_model.zip",
    )
    parser.add_argument(
        "--slope-policy",
        default="checkpoints/phase2/go1/slope/selected_model.zip",
    )
    parser.add_argument(
        "--stairs-policy",
        default="checkpoints/phase2/go1/stairs/selected_model.zip",
    )
    parser.add_argument("--sim2real-config", default="configs/sim2real/go1.yaml")
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-wtw", action="store_true")
    parser.add_argument("--wtw-root", default="external_tmp/walk_these_ways")
    parser.add_argument("--external-report", default=None)
    parser.add_argument("--output", default="results/go1_head_to_head.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    compare_go1_baselines(
        flat_policy=args.flat_policy,
        slope_policy=args.slope_policy,
        stairs_policy=args.stairs_policy,
        sim2real_config=args.sim2real_config,
        n_eval_episodes=args.n_eval_episodes,
        horizon=args.horizon,
        seed=args.seed,
        include_wtw=args.include_wtw,
        wtw_root=args.wtw_root,
        external_report=args.external_report,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
