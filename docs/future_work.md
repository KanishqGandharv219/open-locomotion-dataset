# Future Work

This document outlines planned OLSD extensions that are architecturally designed but not yet implemented, along with their compute and data requirements.

## Phase 3: Multi-Embodiment Locomotion Diffusion Prior

**Status:** Deferred pending compute access.

**Goal:** Train a conditional DDPM on OLSD v2 cross-robot data to learn a shared locomotion prior that transfers across embodiments.

**Architecture (planned):**
- Predict-then-control split: diffusion model predicts the next kinematic observation (joint positions), then a small residual MLP (≤ 64 units) maps the target observation to joint torques/actions.
- Conditioning: robot ID embedding + terrain type embedding.
- Scale: horizon = 8, channels = 64, batch = 32.
- Training data: Go1 canonical trajectories + GrandTour ANYmal-D slices + TAIL traces, aligned to `max_dof = 12` using the existing `alignment.py` utilities.

**Compute requirement:** Fits in 4 GB VRAM (GTX 1650) or free Colab T4. Estimated training time: 8–12 hours.

**Why deferred:** Phases 1–2 deliver more immediate value as a benchmark and toolkit. The diffusion prior is a research contribution that benefits from having clean, velocity-tracking training data (which Phase 2.1 now produces).

## Phase 4: Native-Environment Cross-Evaluation

**Status:** Planned.

**Goal:** Evaluate external baselines (e.g., walk-these-ways) in their native simulator (Isaac Gym) alongside OLSD's MuJoCo evaluation, and compare only the extracted trajectory CSVs.

This eliminates the current caveat that the head-to-head runs the external checkpoint through an OLSD compatibility adapter. The Environment Delta table in `docs/unitree_comparison.md` documents the specific simulator differences that motivate this work.

**Approach:**
1. Run walk-these-ways natively in Isaac Gym, export joint trajectories to CSV.
2. Run OLSD Go1 natively in MuJoCo, export joint trajectories to CSV.
3. Compare using the existing `alignment_eval.py` metrics (RMSE, DTW, velocity correlation) on the shared physical dimensions only.

## Phase 5: Real Hardware Deployment Validation

**Status:** Aspirational.

**Goal:** Deploy an OLSD-trained Go1 policy on physical Unitree Go1 hardware and measure sim-to-real transfer gap.

**Requirements:**
- Access to Unitree Go1 hardware.
- ROS2 bridge for real-time joint position commands.
- Motion capture system for ground-truth trajectory comparison.

This phase would complete the sim-to-real loop and provide the strongest possible validation of the OLSD pipeline.
