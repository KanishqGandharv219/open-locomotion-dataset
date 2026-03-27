# Contributing to OLSD

Thank you for your interest in contributing to the Open Locomotion Skills Dataset!

## Ways to Contribute

### 1. Submit Locomotion Data

If your lab has robot locomotion trajectories, we'd love to include them.

**Requirements:**
- Data must include joint positions and velocities at minimum
- Must be under a permissive license: **CC0**, **CC-BY-4.0**, or **MIT**
- Include robot URDF/MJCF if possible

**Process:**
1. Fork this repository
2. Create a directory: `data/submissions/<your_robot_name>/`
3. Add your data files (HDF5, NumPy, or CSV format)
4. Add a `metadata.yaml` with robot and trajectory info
5. Run validation: `olsd validate data/submissions/<your_robot_name>/`
6. Submit a Pull Request

### 2. Add New Robot Configs

Create a YAML file in `configs/robots/` following the existing format (see `go1.yaml` as template).

### 3. Code Contributions

- Fork → branch → implement → test → PR
- Run `pytest tests/ -v` before submitting
- Run `ruff check .` for linting

### 4. Bug Reports & Feature Requests

Open a GitHub Issue with a clear description and reproduction steps.

## Data Licensing Policy

All contributed data must use one of:
- **CC0** (public domain) — preferred
- **CC-BY-4.0** (attribution required)
- **MIT License**

Proprietary or restricted-use data cannot be included in OLSD.

## Code of Conduct

Be respectful, constructive, and collaborative. We follow standard open-source community guidelines.
