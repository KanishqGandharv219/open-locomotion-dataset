# Changelog

All notable changes to the Open Locomotion Skills Dataset are documented here.

## [2.0.0] - 2026-04-01

### Added

- **Velocity-tracking reward** — Go1 flat policy walks at 0.61 m/s (12m displacement), replacing the stability-only reward that produced near-zero forward velocity
- **Per-episode velocity randomization** — Training samples target velocity from [0.3, 0.8] m/s for robust terrain generalization
- **Sim-to-real toolkit** (`olsd/sim2real/`) — System ID (CMA-ES, 8 scalars), domain randomization configs, alignment evaluation
- **Procedural terrain generation** — Flat, slope, and stairs with configurable parameters
- **ANYmal-D real-data calibration** — GrandTour kinematics ingested with 0.9997 velocity correlation
- **External dataset ingestors** — GrandTour (ANYmal-D), TAIL (deformable terrain), Unitree G1 (retargeted motions) with license gating
- **Cross-embodiment alignment** — Training-time padding, active-dimension masking, per-robot normalization
- **Head-to-head benchmark harness** — Strict shared-metric protocol (success rate, falls, velocity) excluding reward
- **walk-these-ways adapter** — Native JIT checkpoint loading for the pretrain-v0 Go1 baseline
- **Environment Delta documentation** — Explicit MuJoCo vs Isaac Gym comparison table
- **Future work roadmap** — Phase 3 (diffusion prior), Phase 4 (native cross-eval), Phase 5 (real hardware)
- **HuggingFace release** — Trained models and configs at `kanishqgandharv/olsd-v2.0`

### Changed

- **Reward function** — Switched from alive_bonus + clipped velocity to exponential velocity-tracking (`exp(-4 * error²)`) with z-velocity, torque, and action-rate penalties
- **Schema** — Added provenance fields, sim-to-real alignment fields, velocity command tracking to EpisodeMetadata
- **Morphology enum** — Added WHEELED_BIPED
- **Landing page** — Updated to showcase 0.61 m/s velocity and multi-robot support

### Benchmark Results

| Terrain | Success Rate | Forward Velocity | Falls |
|---|---|---|---|
| Flat | 1.0 | 0.61 m/s | 0 |
| Stairs | 0.8 | 0.09 m/s | 4 |
| Slope | 0.0 (in progress) | — | — |

## [0.1.0] - 2026-03-19

### Added

- Initial OLSD schema with Pydantic v2 (Episode, Step, Observation, Action)
- MuJoCo Go1 environment with position control
- Basic PPO training pipeline with Stable-Baselines3
- Hopper/HalfCheetah/Walker2d/Ant demo environments
- LeRobot v3 Parquet export
- Landing page with Three.js background
