# OLSD — Open Locomotion Skills Dataset

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**A unified, open-source benchmark for legged robot locomotion across diverse morphologies and terrains.**

OLSD aims to become the **ImageNet of robot locomotion** — a foundational shared resource that accelerates research, enables cross-lab comparisons, and democratizes access to high-quality training data for legged robots.

---

## Features

- **Unified Schema** — Pydantic-validated trajectory format compatible with LeRobot v3 and RLDS
- **Multi-Robot Support** — HalfCheetah, Ant, Walker2d, Hopper, Unitree Go1, ANYmal, and more
- **Data Pipeline** — Ingest from HDF5, NumPy, CSV, ROS bags, and Gymnasium environments
- **Trajectory Generator** — MuJoCo-based generation with domain randomization
- **Standardized Rewards** — Configurable walking and terrain traversal reward functions
- **Gait Metrics** — Auto-compute stride frequency, energy efficiency, smoothness, and more
- **Visualization** — Dark-themed plots for trajectories, phase portraits, gait diagrams
- **CLI** — Generate, validate, export, and upload datasets from the terminal
- **HF Integration** — Native Hugging Face Datasets support for streaming and versioning

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

# View dataset info
olsd info ./data/generated

# Validate a dataset
olsd validate ./data/generated

# Compute gait metrics
olsd metrics ./data/generated

# Export to different format
olsd export ./data/generated --format hdf5 --output ./data/export

# Upload to Hugging Face
olsd upload ./data/generated --repo your-org/olsd-v0.1
```

## Data Schema

Each **Episode** contains:

| Field | Type | Description |
|-------|------|-------------|
| `observation.joint_positions` | `float[]` | Joint angles (rad) |
| `observation.joint_velocities` | `float[]` | Joint velocities (rad/s) |
| `observation.joint_torques` | `float[]?` | Applied torques (N·m) |
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

*Built by Kanishq Gandharv — Making robot locomotion research open and reproducible.*
