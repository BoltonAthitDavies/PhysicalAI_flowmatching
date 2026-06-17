# PhysicalAI Flow Matching — Pose Trajectory Generation

Conditional trajectory generation with **T-CFM (Trajectory Conditional Flow Matching)** for 2D / 3D pose trajectories under physical constraints (trapezoidal velocity profiles, rotate-first-then-translate motion).

This repo trains a family of CFM models with shared 1D-UNet + FiLM architecture, varying only in **output dimension** (1D θ, 2D xy, 3D xyθ) and **context layout** (which physical features are conditioned on). All models output 50-timestep waypoint trajectories that respect the conditioning context.

---

## Repository Layout

```
config/         T-CFM training configs (cfm_pose.py is the main one)
data/           Dataset .npz archives — one per training variant
diffuser/       CFM library: temporal_film (UNet), cfm.py (sampler), trainer
logs/           Trained checkpoints — one subdir per model variant
scripts/
  train.py                                 Train a new model
  trajectory_generator3Dmultipleinit_npz.py  Synthetic data generator
  evaluation_context.py                     Single-model deployment (ROS controller)
  evaluation_context_model3.py              Composite rotate→translate controller
  traj_generator_eval.ipynb                 Multi-model evaluation notebook
  evaluate_pose_*.ipynb                     Per-variant exploration notebooks
```

---

## Model Family

All variants share the same backbone (`ConditionalUnet1D` + `CFM`, `dim_mults=(1,4,8)`, `HORIZON=64`, `N_SAMPLING_STEPS=10` at inference). They differ only in `transition_dim` (output) and `global_cond_dim` (context).

| Model | Output | Context dim | Dataset (training) | Checkpoint dir |
|---|---|---|---|---|
| **athit3D** | (50, 3) — x, y, θ | 11 | `pose_traject_dataset_3D.npz` | `logs/pose_trajectory_3D/.../20260430-1155/` |
| **athit3D multiinit** | (50, 3) — x, y, θ | 13 (+ω, +α) | `pose_traject_dataset_3D_multiinit.npz` | `logs/pose_trajectory_3D_multiinit/.../20260512-1337/` |
| **athit_theta** (model 3a) | (50, 1) — θ only | 5 | `pose_traject_dataset_athit_theta.npz` | `logs/pose_trajectory_athit_theta/.../20260428-1149/` |
| **athit_notrajpart** (model 3b) | (50, 2) — x, y | 8 | `pose_traject_dataset_athitv2.npz` (re-encoded) | `logs/pose_trajectory_athit_notrajpart/.../20260416-1405/` |

**Model 3 (composite)** chains the last two: a θ-only model handles in-place rotation to align heading, then an xy-only model handles straight-line translation. Each sub-model is invoked sequentially by the controller in [`scripts/evaluation_context_model3.py`](scripts/evaluation_context_model3.py).

### Context-vector layouts per model

```
athit3D (11-dim):
  [s_goal_x, s_goal_y, s_goal_θ, v_const, accel,
   q_init_x, q_init_y, q_init_θ, qdot_init_x, qdot_init_y, qdot_init_θ]

athit3D multiinit (13-dim):
  [...above 11, plus omega_const, alpha_const inserted after accel]

athit_theta (5-dim, rotation sub-model):
  [s_goal_θ, v_const, accel, q_init_θ, qdot_init_θ]

athit_notrajpart (8-dim, xy sub-model):
  [s_goal_x, s_goal_y, qdot_init_x, qdot_init_y, v_const, accel, q_init_x, q_init_y]
```

---

## Dataset Variants

All datasets follow the same `.npz` schema:

```python
data = np.load("pose_traject_dataset_*.npz", allow_pickle=True)
data['features']       # (N, F)   — context features per window
data['targets']        # (N, 50, D) — D=1 (θ), 2 (xy), or 3 (xyθ)
data['feature_names']  # (F,)     — column names
```

| File | F | D | Targets meaning | Used by |
|---|---|---|---|---|
| `pose_traject_dataset_3D.npz` | 12 | 3 | x, y, θ | athit3D |
| `pose_traject_dataset_3D_multiinit.npz` | 14 | 3 | x, y, θ | athit3D multiinit |
| `pose_traject_dataset_athit_theta.npz` | 7 | 1 | θ | athit_theta (model 3a) |
| `pose_traject_dataset_athitv2.npz` | 14 | 7 | x, y, z, quat — only x,y used | athit_notrajpart (model 3b) |
| `pose_traject_dataset_athit*.npz` | various | 2 | x, y | older 2D variants |

Each window is a 50-step slice of a longer trapezoid trajectory at **25 Hz** (2.0 s per window). Windows starting at `t_init = 0` cover the full motion; later starts give the model "mid-motion" continuations.

### Dataset generation
[`scripts/trajectory_generator3Dmultipleinit_npz.py`](scripts/trajectory_generator3Dmultipleinit_npz.py) synthesises trapezoid-profile trajectories with **rotate-first-then-translate** semantics:
1. **Rotation phase** — in-place trapezoid in θ from `init_θ` to `theta_goal` using `(omega_max, alpha_max)`.
2. **Translation phase** — straight-line trapezoid in xy from `init` to `goal` using `(v_max, a_max)`.
3. **End hold** — 50 extra samples at the goal pose so late windows have valid 50-step ground truth.

Sampling ranges (matches the evaluation spec):

| Variable | Min | Max | Notes |
|---|---|---|---|
| `init_x`, `init_y` | -5.0 | 5.0 | metres |
| `init_theta` | -π | π | rad |
| `r_goal` (init→goal distance) | 0.0 | 5.0 | metres |
| `theta_goal` | -π | π | rad |
| `v_max` | 0.10 | 0.20 | m/s |
| `a_max` | 0.02 | 0.04 | m/s² |
| `omega_max` | 0.5 | 1.0 | rad/s |
| `alpha_max` | 0.05 | 0.10 | rad/s² |

Stopping is **time-based** from closed-form trapezoid math — no error-tolerance check. By construction the trajectory ends at the goal with zero velocity.

---

## T-CFM Architecture (shared backbone)

```
                Input: noisy trajectory x_t  ─── (B, 64, D)
                                  │
              rearrange (B, D, 64)  ← channels-first for Conv1d
                                  │
┌─── CONDITIONING ──────────────────────────────────┐
│  Flow time t ──► SinusoidalPosEmb ──► MLP ─► 32  │
│  Context C    ────────────────────────────────┐  │
│                                       concat  ▼  │
│                              global_feature (B, 32+ctx_dim)
└──────────────────────────────────┼────────────────┘
                                  │
                          ┌───────▼───────┐
                          │  DOWN (×3)    │   channels 32 → 128 → 256
                          │  Block + Down │
                          ├───────────────┤
                          │  MID (×2)     │
                          ├───────────────┤
                          │  UP (×3)      │   skip connections from DOWN
                          │  Block + Up   │
                          ├───────────────┤
                          │  Final Conv   │  → D channels
                          └───────┬───────┘
                                  │
                rearrange to (B, 64, D)  → predicted vector field v_θ
```

- **`dim_mults=(1, 4, 8)`** → channel sizes `[32, 128, 256]`. 2 downsamples → horizon must be divisible by 4 (50 padded to 64).
- **FiLM conditioning**: `global_feature` modulates every `ConditionalResidualBlock1D` via a learned bias on each conv output.
- **~2.35 M parameters** for the 2D variant; minor scaling for D=1 / D=3 final-conv channel counts.

### Training objective (CFM loss)

```python
x1 = real_trajectory                  # (B, 64, D) — ground truth
x0 = torch.randn_like(x1)             # (B, 64, D) — pure Gaussian noise
t  = uniform(0, 1)                    # scalar per sample
xt = t * x1 + (1 - t) * x0            # linearly interpolated
ut = x1 - x0                          # straight target flow

vt   = model(t, xt, global_cond=context)
loss = mean((vt - ut) ** 2)
```

### Inference (Euler ODE)

```
x_0 ~ N(0, I)                                   shape: (B, 64, D)
for k = 0 .. N-1:
    x_{k+1} = x_k + (1/N) · v_θ(k/N, x_k, C)
return x_N[:, :50, :]                           shape: (B, 50, D)
```

Default `N_SAMPLING_STEPS = 10` (CFM converges in far fewer steps than diffusion; even N=1 produces sensible results).

---

## Evaluation Workflow

The primary evaluation notebook is [`scripts/traj_generator_eval.ipynb`](scripts/traj_generator_eval.ipynb). It does the following end-to-end:

1. **Generates fresh evaluation data** via `create_dataset()` from `trajectory_generator3Dmultipleinit_npz.py` (configurable `EVAL_N_TRAJECTORIES`, fixed seed). Keeps only `t_init == 0` windows so one trajectory = one sample (no overlapping slices).
2. **Loads 4 sub-models** (athit3D, athit3D multiinit, athit_theta, athit_notrajpart) and routes the right context slice to each.
3. **Inpainted Euler sampler** — `cfm_sample_inpaint` runs a manual Euler loop and overwrites `x[:, 0, :] = q_init` after each step. This anchors every generated trajectory at the ground-truth init pose, matching deployment behaviour (the controllers in `evaluation_context*.py` always re-sample from the current pose).
4. **Comparison plots** (6-panel per sample): `x, y, θ` and their finite-difference derivatives `ẋ, ẏ, θ̇`, overlaying ground truth and all three models.
5. **Aggregate error analysis**:
   - Per-channel MAE table + quantile summary (p50, p90, p95, max).
   - Per-window MAE distribution: histograms + boxplots per channel.
   - Scatter plots of MAE vs each of 14 context features, with linear trend lines.
   - **Three correlation heatmaps stacked**: Pearson r (linear), Spearman ρ (monotonic), Mutual Information (any dependence — surfaces non-monotonic relationships invisible to ρ).

### Key implementation note: first-timestep inpainting

Vanilla `CFM.conditional_sample` ignores the `cond` argument inside `p_sample_loop_cfm` ([diffuser/models/cfm.py:405](diffuser/models/cfm.py#L405)) — it just integrates from random noise. Generated trajectories therefore *drift* from the conditioned `q_init` at t=0. The evaluation notebook bypasses this by manually integrating with hard-anchored x[0]; the existing deployment scripts inherit the drift (they re-plan often enough that it does not matter in practice).

### Composite Model 3 — rotate-then-translate

[`scripts/evaluation_context_model3.py`](scripts/evaluation_context_model3.py) deploys the composite controller used in TurtleSim:
- On new `/goal_position`, sample a θ trajectory from the **athit_theta** model and PID-track it until heading aligns with `atan2(dy, dx)`.
- Switch to the **athit_notrajpart** model: sample an xy trajectory from the current pose and PID-track waypoints at 25 Hz, re-sampling whenever the 50-step buffer is exhausted.

---

## Training a New Model

```bash
python scripts/train.py --config config.cfm_pose
```

The dataset path and context layout are currently configured inline in [`scripts/train.py`](scripts/train.py) — uncomment the relevant block for the variant you want to train. Checkpoints land in `logs/<variant>/cfm/H64_T100/<timestamp>/` and a state is saved every 5 000 steps (200 000 total by default).

---

## Quick Start: Evaluate Existing Models

```bash
# 1. Verify checkpoints are in place
ls logs/pose_trajectory_3D/cfm/H64_T100/20260430-1155/state_192000.pt
ls logs/pose_trajectory_3D_multiinit/cfm/H64_T100/20260512-1337/state_192000.pt
ls logs/pose_trajectory_athit_theta/cfm/H64_T100/20260428-1149/state_192000.pt
ls logs/pose_trajectory_athit_notrajpart/cfm/H64_T100/20260416-1405/state_192000.pt

# 2. Open the evaluation notebook
jupyter notebook scripts/traj_generator_eval.ipynb
# (or open in VS Code)

# 3. Adjust EVAL_N_TRAJECTORIES in cell 4 and run all cells.
```

The notebook produces (a) per-sample 6-panel trajectory plots, (b) per-channel MAE summary, (c) error-distribution histograms / boxplots, (d) feature-vs-error scatters with trend lines, and (e) Pearson + Spearman + MI heatmaps for diagnosing which context features drive each model's error.
