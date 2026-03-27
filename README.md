# OLSD -- Open Locomotion Skills Dataset

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/KanishqGandharv219/open-locomotion-dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/KanishqGandharv219/open-locomotion-dataset/actions)

**A unified, open-source benchmark for legged robot locomotion across diverse morphologies and terrains.**

OLSD aims to become the **ImageNet of robot locomotion** -- a foundational shared resource that accelerates research, enables cross-lab comparisons, and democratizes access to high-quality training data for legged robots.

---

## Features

- **Unified Schema** -- Pydantic-validated trajectory format compatible with LeRobot v3 and RLDS
- **Multi-Robot Support** -- HalfCheetah, Ant, Walker2d, Hopper, Unitree Go1, and more
- **Data Pipeline** -- Ingest from HDF5, NumPy, CSV, ROS bags, and Gymnasium environments
- **Trajectory Generator** -- MuJoCo-based generation with domain randomization
- **RL Baselines** -- PPO and SAC training with D4RL-style normalized evaluation
- **Gait Metrics** -- Auto-compute stride frequency, energy efficiency, smoothness, and more
- **Visualization** -- Trajectory plots, phase portraits, gait diagrams, reward signals
- **CLI** -- Generate, train, evaluate, validate, export datasets from the terminal
- **HF Integration** -- Native Hugging Face Datasets support for streaming and versioning

## Benchmark Results

Trained baselines evaluated with D4RL-style normalized scores (0 = random, 100 = expert):

| Robot | Algorithm | Return | Norm. Score | Success Rate | Ep. Length |
|-------|-----------|--------|-------------|-------------|------------|
| HalfCheetah | PPO (1M steps) | 1955.4 | 18.0 | 100.0% | 1000 |
| Ant | PPO (2M steps) | 1733.2 | 29.7 | 80.0% | 892 |
| Walker2d | SAC (1M steps) | 4928.5 | **107.3** | 90.0% | 919 |
| Hopper | PPO (500K steps) | 3507.9 | **108.4** | 100.0% | 1000 |

> Walker2d and Hopper achieve superhuman performance (normalized score > 100).

## Dataset Stats

| Metric | Value |
|--------|-------|
| Total Episodes | ~4,000 |
| Total Steps | ~2.5M |
| Robots | HalfCheetah, Ant, Walker2d, Hopper |
| Quality Tiers | random, expert, domain_random |
| Format | Parquet (LeRobot v3-compatible) |

## Installation

```bash
pip install -e .
```

With optional dependencies:
```bash
pip install -e ".[all]"     # Everything (ROS bags, dev tools)
pip install -e ".[dev]"     # Development (pytest, ruff)
pip install -e ".[rosbag]"  # ROS bag support
```

## Quick Start

### Python SDK

```python
import olsd

# Generate trajectories
from olsd.generation.mujoco_gen import generate_trajectories
episodes = generate_trajectories("halfcheetah", n_episodes=50, policy="random")

# Export to Parquet
from olsd.pipeline.export import to_parquet
to_parquet(episodes, "./data/my_dataset")

# Load and filter
dataset = olsd.load("./data/my_dataset")
print(dataset.summary())

# Visualize
from olsd.sdk.visualization import plot_trajectory
plot_trajectory(dataset[0])
```

### CLI

```bash
# Generate trajectories
olsd generate --robot halfcheetah --robot ant --episodes 100 --output ./data/generated

# Train RL baselines
olsd train -r halfcheetah -a ppo -t 1000000
olsd train -r walker2d -a sac -t 1000000

# Evaluate with D4RL-style scores
olsd eval --all -n 50

# View dataset info
olsd info ./data/generated

# Validate a dataset
olsd validate ./data/generated

# Compute gait metrics
olsd metrics ./data/generated

# Export to different format
olsd export ./data/generated --format hdf5 --output ./data/export
```

See [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) for an interactive walkthrough.

## Data Schema

Each **Episode** contains:

| Field | Type | Description |
|-------|------|-------------|
| `observation.joint_positions` | `float[]` | Joint angles (rad) |
| `observation.joint_velocities` | `float[]` | Joint velocities (rad/s) |
| `observation.joint_torques` | `float[]?` | Applied torques (Nm) |
| `observation.imu_orientation` | `float[4]?` | Quaternion [w,x,y,z] |
| `observation.contact_forces` | `float[]?` | Ground reaction forces |
| `action` | `float[]` | Motor commands |
| `reward` | `float?` | Reward signal |
| `done` | `bool` | Terminal flag |

Each episode includes metadata: robot spec, terrain type, gait, speed, energy cost, data source, and attribution.

## Supported Robots

| Robot | Morphology | Joints | Source |
|-------|-----------|--------|--------|
| HalfCheetah | Planar | 6 | MuJoCo/Gymnasium |
| Ant | Quadruped | 8 | MuJoCo/Gymnasium |
| Walker2d | Biped | 6 | MuJoCo/Gymnasium |
| Hopper | Monoped | 3 | MuJoCo/Gymnasium |
| Unitree Go1 | Quadruped | 12 | MuJoCo Menagerie |

## Project Structure

```
olsd/
├── schema/          # Pydantic models (trajectory, metadata, rewards)
├── pipeline/        # Ingest, validate, compute metrics, export
├── generation/      # MuJoCo trajectory generation + domain randomization
├── sdk/             # Dataset loader + visualization
├── benchmark/       # Baseline training + evaluation
└── cli.py           # Command-line interface
```

## License

Apache 2.0. Datasets may carry their own licenses (CC-BY-4.0, CC0, MIT).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting data, code, and bug reports.

---

*Built by Kanishq Gandharv -- Making robot locomotion research open and reproducible.*
