# OLSD v2 vs Unitree-Focused Baseline

This document answers a narrower question than marketing copy: based on the artifacts currently verified in this repository, where does OLSD v2 already beat a Unitree-focused baseline, and where is it still unproven?

## Short Answer

OLSD v2 is already stronger as an open benchmark and sim-to-real toolkit.

OLSD v2 now also has a direct strict-protocol comparison against one named external Go1 baseline: `walk-these-ways` `pretrain-v0`.

Under the OLSD shared-metric protocol, OLSD v2 wins clearly on stability across flat, slope, and stairs. That result is real and reproducible in this repo.

OLSD v2 is still not proven universally superior in raw locomotion performance, because the external baseline is being executed through an OLSD compatibility adapter in MuJoCo, not through its original native Isaac Gym evaluation stack.

## What Is Verified In This Repo

The comparison below is grounded in:

- `71` passing tests
- canonical Go1 Phase 2 artifacts under `checkpoints/phase2/go1/`
- real-data-backed ANYmal-D alignment artifacts under `configs/sim2real/anymal_d.yaml` and `configs/sim2real/anymal_d.alignment.json`
- the consolidated report at `results/sim2real_report.json`
- the direct Go1 head-to-head report at `results/go1_head_to_head.json`

## Comparison Table

| Dimension | OLSD v2 today | Unitree-focused baseline | Verdict |
| --- | --- | --- | --- |
| Robot coverage | Verified Go1 policies plus real ANYmal-D alignment, with external ingestors for GrandTour, TAIL, and Unitree retargeted motion | Typically centered on one vendor family or one robot class | OLSD leads on scope |
| Terrain coverage | Canonical Go1 outputs for flat, slope, and stairs, plus GrandTour-derived ANYmal multi-terrain real data | Often narrower or task-specific unless expanded manually | OLSD leads on terrain diversity |
| Data format | Unified open schema with sim, hardware, provenance, and training-time alignment support | Usually task-specific or vendor-specific formats | OLSD leads on openness |
| Sim-to-real tooling | System ID, alignment eval, terrain generation, domain-randomization configs, and closeout report are in-repo | Usually scattered across demos, papers, or private workflows | OLSD leads on reproducibility |
| Real-data grounding | ANYmal-D closeout is backed by local GrandTour hardware slices | Varies by specific Unitree dataset or stack | OLSD has verified real-data evidence in this repo |
| Canonical artifacts | Release-style selected models and summary files are checked in for Go1 | Often checkpoint-specific or demo-specific | OLSD leads on artifact hygiene |
| Cross-vendor benchmark value | Vendor-neutral | Vendor-specific by definition | OLSD leads |
| Direct Go1 head-to-head in this repo | `results/go1_head_to_head.json` shows stable OLSD runs on all three terrains | `walk-these-ways` `pretrain-v0` executes through a native adapter but falls early in the OLSD protocol | OLSD leads under the OLSD protocol |
| Raw locomotion performance in each stack's native benchmark | Not yet measured side-by-side | Not yet measured side-by-side | Still unproven |

## Verified OLSD Numbers

### Go1 Canonical Policies

| Terrain | Selected source | Return mean | Episode length mean | Success rate |
| --- | --- | ---: | ---: | ---: |
| Flat | `best_checkpoint` | `985.70` | `1000.0` | `1.0` |
| Slope | `init_policy` | `965.00` | `1000.0` | `1.0` |
| Stairs | `init_policy` | `965.00` | `1000.0` | `1.0` |

### ANYmal-D Real Alignment

| Metric | Value |
| --- | ---: |
| Joint RMSE | `0.0161` |
| Velocity correlation | `0.9997` |
| Trajectory DTW | `0.0259` |
| Episode pairs | `3` |
| Shared steps | `1024` |

### Direct Go1 Head-To-Head

This repo now includes a strict shared-metric comparison against the public `Improbable-AI/walk-these-ways` Go1 `pretrain-v0` checkpoint.

The comparison uses:

- same robot: Go1
- same terrains: flat, slope, stairs
- same horizon: `1000`
- same episode count: `20`
- same reported metrics: success rate, episode length, forward velocity, and falls
- no reward comparison

| Baseline | Terrain | Success rate | Episode length mean | Forward velocity mean | Fall count |
| --- | --- | ---: | ---: | ---: | ---: |
| OLSD v2 canonical | Flat | `1.0` | `1000.0` | `0.0048` | `0` |
| OLSD v2 canonical | Slope | `1.0` | `1000.0` | `0.0052` | `0` |
| OLSD v2 canonical | Stairs | `1.0` | `1000.0` | `0.0052` | `0` |
| walk-these-ways pretrain-v0 | Flat | `0.0` | `33.0` | `0.1219` | `20` |
| walk-these-ways pretrain-v0 | Slope | `0.0` | `31.0` | `0.1429` | `20` |
| walk-these-ways pretrain-v0 | Stairs | `0.0` | `31.0` | `0.1429` | `20` |

Interpretation:

- OLSD is much more stable under the OLSD MuJoCo benchmark.
- The adapted `walk-these-ways` baseline still produces forward motion, so the adapter is not degenerate.
- But it does not transfer stably into this benchmark as currently configured.

### Environment Delta

The following table documents the known differences between the OLSD Go1 evaluation environment and the native `walk-these-ways` training environment. Performance differences in the head-to-head may reflect these environmental deltas, not pure policy quality.

| Property | OLSD v2 | walk-these-ways |
| --- | --- | --- |
| Simulator | MuJoCo 3.x | Isaac Gym (PhysX GPU) |
| Physics backend | MuJoCo native contact solver | NVIDIA PhysX |
| Go1 model file | Hand-authored MJCF (`go1_env.py`) | URDF from `legged_gym` |
| Observation space | 35-dim (qpos + qvel) | 70-dim (+ observation history + commands) |
| Action space | 12-dim position targets | 12-dim position offsets from default pose |
| Control frequency | 50 Hz (0.002s × 10 frame skip) | 50 Hz |
| Reward function | OLSD velocity-tracking (exp-form) | Gait-conditioned multi-objective |
| Training framework | Stable-Baselines3 PPO (CPU) | rl_games PPO (GPU) |

## Where OLSD v2 Clearly Wins

- Open, vendor-neutral schema instead of a single-vendor workflow
- Verified multi-terrain benchmark outputs instead of only one demonstration path
- Real-data-backed ANYmal alignment in the same repo as the Go1 training pipeline
- Canonical selected-model logic that preserves transferred policies when fine-tuning gets worse
- A single source-of-truth Phase 2 report instead of scattered notebooks or logs
- A direct reproducible Go1 head-to-head now exists in this repo, and OLSD wins it on shared stability metrics

## Where We Should Stay Honest

- We cannot yet claim OLSD v2 has better locomotion control in every benchmark context.
- The current direct benchmark runs `walk-these-ways` through an OLSD MuJoCo compatibility adapter, not its original Isaac Gym stack.
- The current ANYmal-D identification path uses the reduced kinematic replay backend, not a full actuator-level physics replay backend.

## Defensible Claim Today

The honest claim from the verified artifacts is:

> OLSD v2 is better as an open, multi-robot, multi-terrain benchmark and sim-to-real toolkit. Under the strict OLSD Go1 shared-metric protocol, OLSD v2 also outperforms the integrated `walk-these-ways` `pretrain-v0` baseline on stability across flat, slope, and stairs. That does not yet prove universal superiority over every Unitree or Isaac Gym evaluation setup.
