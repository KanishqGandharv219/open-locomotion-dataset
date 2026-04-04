# OLSD v2

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/KanishqGandharv219/open-locomotion-dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/KanishqGandharv219/open-locomotion-dataset/actions)
[![Website](https://img.shields.io/badge/Website-Live-8b7cf6)](https://kanishqgandharv219.github.io/open-locomotion-dataset/)

Open Locomotion Skills Dataset is a unified locomotion data and sim-to-real toolkit for legged robots. OLSD v2 adds external hardware ingestors, training-time embodiment alignment utilities, a reusable Go1 training environment, reduced-parameter system ID, domain-randomization configs, and canonical Phase 2 sim-to-real artifacts.

## Verified Status

As of March 30, 2026, the repository is verified at:

- `71` passing tests via `python -m pytest tests -q`
- Phase 1 complete: schema upgrades, external ingestors, alignment utilities
- Phase 2 complete: Go1 sim-to-real kit, terrain generation, ANYmal-D real-data alignment, final closeout report
- strict Go1 head-to-head completed with an integrated `walk-these-ways` `pretrain-v0` baseline adapter

Canonical Phase 2 artifacts live under:

- `checkpoints/phase2/go1/flat/selected_model.zip`
- `checkpoints/phase2/go1/slope/selected_model.zip`
- `checkpoints/phase2/go1/stairs/selected_model.zip`
- `checkpoints/phase2/go1/*/summary.json`
- `configs/sim2real/go1.yaml`
- `configs/sim2real/anymal_d.yaml`
- `configs/sim2real/anymal_d.alignment.json`
- `results/sim2real_report.json`
- `results/go1_head_to_head.json`
- `results/go1_head_to_head_schema.json`

For a conservative comparison against a Unitree-focused baseline, see `docs/unitree_comparison.md`.

## What Is In v2

- Unified Pydantic schema for locomotion trajectories and metadata
- External data ingestion for GrandTour, TAIL, and Unitree retargeted motion
- Training-time cross-embodiment alignment utilities
- Reusable Go1 MuJoCo environment and PPO training script
- Reduced-parameter sim-to-real identification with CMA-ES
- Per-robot domain-randomization config generation
- Procedural flat, slope, and stairs terrain generation
- Canonical sim-to-real artifact layout for release-style checkpoints

## Phase 2 Results

These numbers come from the checked-in artifacts and `results/sim2real_report.json`.

### Go1 Canonical Policies (Velocity-Tracking v1):

| Terrain | Status | Return Mean | Episode Length Mean | Forward Velocity | Success Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Flat | **canonical** | `2088.6` | `1000.0` | `0.61 m/s` | `1.0` |
| Stairs | **canonical** | `1006.9` | `920.6` | `0.09 m/s` | `0.8` |
| Slope | *in progress* | `248.0` | `228.8` | — | `0.0` |

Notes:

- Flat policy uses exponential velocity-tracking reward with per-episode velocity randomization from `[0.3, 0.8]` m/s during training.
- Stairs uses the flat policy as warm-start (init_policy), which outperformed dedicated fine-tuning.
- Slope requires longer training (>2M steps) or terrain-specific reward tuning; deferred to v2.0.1.
- The flat policy covers **12 meters** in 1000 steps at **0.60 m/s** average, verified independently.

### ANYmal-D Real Alignment

The ANYmal-D closeout used a local GrandTour subset materialized into three `50 Hz` calibration slices of `1024` steps each.

| Metric | Value |
| --- | ---: |
| Joint RMSE | `0.0161` |
| Velocity Correlation | `0.9997` |
| Trajectory DTW | `0.0259` |
| Episode Pairs | `3` |
| Shared Steps | `1024` |
| Shared Joints | `12` |

Important caveat:

- The ANYmal-D artifact is real-data-backed from GrandTour and uses the local staged asset at `assets/anymal_d/anymal_d.xml`.
- The current backend is still the reduced kinematic replay backend, not a full actuator-level physics replay backend. The generated config says this explicitly in `configs/sim2real/anymal_d.yaml`.

### Go1 Head-To-Head

The strict shared-metric head-to-head in `results/go1_head_to_head.json` evaluates the velocity-tracking canonical policies:

| Baseline | Terrain | Success Rate | Episode Length Mean | Forward Velocity Mean | Fall Count |
| --- | --- | ---: | ---: | ---: | ---: |
| OLSD v2 veltrack | Flat | `1.0` | `1000.0` | `0.6108` | `0` |
| OLSD v2 veltrack | Stairs | `0.8` | `874.2` | `0.0893` | `4` |

Important caveat:

- These are velocity-tracking policies trained with per-episode velocity randomization.
- Slope is excluded from canonical v2.0 results and will be revisited in v2.0.1.
- The flat policy demonstrates genuine locomotion: 12m displacement at 0.6 m/s average.

## Installation

Core install:

```bash
pip install -e .
```

Recommended for Phase 2 work:

```bash
pip install -e ".[sim,external,dev]"
```

Optional groups:

```bash
pip install -e ".[sim]"
pip install -e ".[external]"
pip install -e ".[dev]"
pip install -e ".[rosbag]"
pip install -e ".[all]"
```

## Reproduce Phase 2

### 1. Run the test suite

```bash
python -m pytest tests -q
```

### 2. Train the canonical Go1 flat policy

```bash
python scripts/train_go1.py \
  --terrain flat \
  --timesteps 200000 \
  --n-envs 4 \
  --batch-size 256 \
  --output-dir checkpoints/go1-flat-sanity-v3 \
  --sim2real-config configs/sim2real/go1.yaml \
  --seed 0
```

### 3. Warm-start non-flat Go1 policies

Slope:

```bash
python scripts/train_go1.py \
  --terrain slope \
  --timesteps 25000 \
  --n-envs 1 \
  --batch-size 64 \
  --output-dir checkpoints/go1-slope-callback-warmstart-v2 \
  --sim2real-config configs/sim2real/go1.yaml \
  --init-policy checkpoints/go1-flat-sanity-v3/flat/best_model/best_model.zip \
  --seed 0
```

Stairs:

```bash
python scripts/train_go1.py \
  --terrain stairs \
  --timesteps 25000 \
  --n-envs 1 \
  --batch-size 64 \
  --output-dir checkpoints/go1-stairs-callback-warmstart-v1 \
  --sim2real-config configs/sim2real/go1.yaml \
  --init-policy checkpoints/go1-flat-sanity-v3/flat/best_model/best_model.zip \
  --seed 0
```

### 4. License-check GrandTour

```bash
python scripts/license_check.py leggedrobotics/grand_tour_dataset
```

### 5. Materialize the ANYmal-D GrandTour subset

```bash
python scripts/materialize_grandtour_anymal.py \
  --source-dir data/external/grandtour_subset \
  --output-dir data/external/grandtour_anymal \
  --max-episodes 3 \
  --target-hz 50 \
  --max-steps 1024
```

### 6. Run ANYmal-D reduced system ID

```bash
python -m olsd.sim2real.system_id \
  --robot anymal_d \
  --real-data data/external/grandtour_anymal \
  --mjcf-path assets/anymal_d/anymal_d.xml \
  --output configs/sim2real/anymal_d.yaml \
  --generations 2 \
  --population 4 \
  --seed 0
```

### 7. Build the final sim-to-real report

```bash
python scripts/build_phase2_report.py \
  --flat-summary checkpoints/phase2/go1/flat/summary.json \
  --slope-summary checkpoints/phase2/go1/slope/summary.json \
  --stairs-summary checkpoints/phase2/go1/stairs/summary.json \
  --anymal-config configs/sim2real/anymal_d.yaml \
  --anymal-alignment configs/sim2real/anymal_d.alignment.json \
  --output results/sim2real_report.json
```

### 8. Run the strict Go1 head-to-head scaffold

This script evaluates the canonical OLSD Go1 policies on shared physical metrics only. It can also load the public `walk-these-ways` `pretrain-v0` checkpoint through the native adapter.

```bash
python scripts/compare_go1_baselines.py \
  --flat-policy checkpoints/phase2/go1/flat/selected_model.zip \
  --slope-policy checkpoints/phase2/go1/slope/selected_model.zip \
  --stairs-policy checkpoints/phase2/go1/stairs/selected_model.zip \
  --sim2real-config configs/sim2real/go1.yaml \
  --n-eval-episodes 20 \
  --horizon 1000 \
  --output results/go1_head_to_head.json
```

To include the public `walk-these-ways` baseline, first clone it locally:

```bash
git clone https://github.com/Improbable-AI/walk-these-ways.git external_tmp/walk_these_ways
```

Then run:

```bash
python scripts/compare_go1_baselines.py \
  --flat-policy checkpoints/phase2/go1/flat/selected_model.zip \
  --slope-policy checkpoints/phase2/go1/slope/selected_model.zip \
  --stairs-policy checkpoints/phase2/go1/stairs/selected_model.zip \
  --sim2real-config configs/sim2real/go1.yaml \
  --include-wtw \
  --wtw-root external_tmp/walk_these_ways \
  --n-eval-episodes 20 \
  --horizon 1000 \
  --output results/go1_head_to_head.json
```

## Repository Layout

```text
olsd/
  pipeline/        ingestion, export, validation, license checks
  schema/          episode, metadata, robot, terrain, and trajectory models
  sdk/             dataset loading and visualization
  sim2real/        system ID, alignment eval, domain configs, terrains, Go1 env

configs/
  robots/          robot specs
  sim2real/        identified params and DR configs

scripts/
  train_go1.py
  materialize_grandtour_anymal.py
  build_phase2_report.py
  license_check.py

checkpoints/phase2/go1/
results/
```

## Current Scope

Completed now:

- Phase 1: schema and external ingestion
- Phase 2: sim-to-real closeout and GitHub-ready artifacts

Not yet shipped:

- Phase 3: multi-embodiment diffusion prior
- Phase 4: release packaging, paper draft, landing page refresh

## License

Code is Apache 2.0. External datasets keep their own licenses.

Verified in this repo:

- GrandTour: permissive and safe to ingest after license check
- Asset and dataset provenance are recorded in the generated artifacts

## Contributing

See `CONTRIBUTING.md` for contribution guidelines.
