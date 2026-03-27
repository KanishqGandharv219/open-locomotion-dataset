"""
OLSD Dataset Assembly — Merge all generated data into one canonical dataset.

Usage:
    python scripts/assemble_dataset.py
"""

import logging
from pathlib import Path

from olsd.pipeline.export import to_parquet
from olsd.sdk.loader import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


DATA_SOURCES = {
    "data/olsd-v0.1":              "random",
    "data/olsd-dr":                "domain_random",
    "data/olsd-expert-halfcheetah": "expert",
    "data/olsd-expert-ant":         "expert",
    "data/olsd-expert-walker2d":    "expert",
    "data/olsd-expert-hopper":      "expert",
}

OUTPUT_DIR = "data/olsd-v0.1-final"


def main():
    all_episodes = []

    for data_path, quality_tier in DATA_SOURCES.items():
        path = Path(data_path)
        if not path.exists():
            logger.warning(f"Skipping {data_path} (not found)")
            continue

        logger.info(f"Loading {data_path} (tier={quality_tier})...")
        dataset = load(str(path))

        for ep in dataset.episodes:
            ep.metadata.quality_tier = quality_tier
            all_episodes.append(ep)

        logger.info(f"  Loaded {len(dataset)} episodes")

    logger.info(f"\nTotal: {len(all_episodes)} episodes")

    # Summary
    from collections import Counter
    robots = Counter(ep.metadata.robot.robot_id for ep in all_episodes)
    tiers = Counter(ep.metadata.quality_tier for ep in all_episodes)
    total_steps = sum(ep.n_steps for ep in all_episodes)

    logger.info(f"Robots: {dict(robots)}")
    logger.info(f"Quality tiers: {dict(tiers)}")
    logger.info(f"Total steps: {total_steps:,}")

    logger.info(f"\nExporting to {OUTPUT_DIR}...")
    to_parquet(all_episodes, OUTPUT_DIR)
    logger.info(f"Done! Canonical dataset at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
