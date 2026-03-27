"""Upload OLSD dataset to Hugging Face Hub."""

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("olsd.upload_hf")


def upload_dataset(
    data_dir: str,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
):
    """Upload a local OLSD dataset directory to Hugging Face Hub.

    Args:
        data_dir: Path to the local dataset directory (e.g., data/olsd-v0.1-final/).
        repo_id: HF repo ID (e.g., 'KanishqGandharv219/olsd-v0.1').
        token: HF API token. If None, uses cached login.
        private: Whether to create a private repo.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    api = HfApi(token=token)

    # Create the repo (dataset type)
    logger.info(f"Creating dataset repo: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    # Build a README for the HF dataset card
    readme = _build_dataset_card(data_path, repo_id)
    readme_path = data_path / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    logger.info("Generated dataset card (README.md)")

    # Upload the entire directory
    logger.info(f"Uploading {data_path} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(data_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload OLSD v0.1 dataset",
        ignore_patterns=["*.pyc", "__pycache__", ".DS_Store", "*.log"],
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"Upload complete! Dataset available at: {url}")
    return url


def _build_dataset_card(data_path: Path, repo_id: str) -> str:
    """Generate a Hugging Face dataset card (README.md)."""

    # Try to read metadata from the dataset
    meta_files = list(data_path.glob("**/metadata.json"))
    stats = {}
    if meta_files:
        try:
            with open(meta_files[0]) as f:
                stats = json.load(f)
        except Exception:
            pass

    total_episodes = stats.get("total_episodes", "~4,200")
    total_steps = stats.get("total_steps", "~2,722,712")

    card = f"""---
license: apache-2.0
task_categories:
  - reinforcement-learning
tags:
  - robotics
  - locomotion
  - mujoco
  - gymnasium
  - benchmark
  - offline-rl
language:
  - en
size_categories:
  - 1M<n<10M
pretty_name: Open Locomotion Skills Dataset (OLSD)
---

# Open Locomotion Skills Dataset (OLSD) v0.1

A unified benchmark dataset for legged robot locomotion across diverse morphologies.

## Dataset Description

OLSD provides expert, random, and domain-randomized trajectories for training and evaluating
locomotion policies. All data is collected from MuJoCo/Gymnasium environments.

| Metric | Value |
|--------|-------|
| Total Episodes | {total_episodes} |
| Total Steps | {total_steps} |
| Robots | HalfCheetah, Ant, Walker2d, Hopper |
| Quality Tiers | random, expert, domain_random |
| Format | Parquet + metadata JSON |

## Benchmark Results

| Robot | Algorithm | Norm. Score | Return |
|-------|-----------|-------------|--------|
| Hopper | PPO | **108.4** | 3,507.9 |
| Walker2d | SAC | **107.3** | 4,928.5 |
| Ant | PPO | 29.7 | 1,733.2 |
| HalfCheetah | PPO | 18.0 | 1,955.4 |

## Usage

```python
from datasets import load_dataset

# Load the full dataset
ds = load_dataset("{repo_id}")

# Or load a specific robot
ds = load_dataset("{repo_id}", data_files="halfcheetah/**/*.parquet")
```

Or with the OLSD SDK:

```python
import olsd
dataset = olsd.load("./data/olsd-v0.1-final")
print(dataset.summary())
```

## Data Schema

Each row contains:
- `observation.joint_positions` -- Joint angles (rad)
- `observation.joint_velocities` -- Joint velocities (rad/s)
- `action` -- Motor commands
- `reward` -- Reward signal
- `done` -- Terminal flag

## Links

- [GitHub](https://github.com/KanishqGandharv219/open-locomotion-dataset)
- [Website](https://kanishqgandharv219.github.io/open-locomotion-dataset/)
- [Quickstart Notebook](https://github.com/KanishqGandharv219/open-locomotion-dataset/blob/main/notebooks/quickstart.ipynb)

## License

Apache 2.0
"""
    return card


def main():
    parser = argparse.ArgumentParser(description="Upload OLSD dataset to Hugging Face Hub")
    parser.add_argument("--data-dir", default="data/olsd-v0.1-final",
                        help="Path to the dataset directory")
    parser.add_argument("--repo-id", default="KanishqGandharv219/olsd-v0.1",
                        help="Hugging Face repo ID (user/dataset)")
    parser.add_argument("--token", default=None,
                        help="HF API token (or use `huggingface-cli login`)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private dataset repo")
    args = parser.parse_args()

    upload_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
