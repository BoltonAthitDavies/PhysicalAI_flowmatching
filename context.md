# T-CFM Model Summary

## 1. Model Input

The model takes the following inputs:

* **Noise trajectory (τ₀):**
  Sampled from a Gaussian distribution, represents the starting point.

* **Target trajectory (τ₁):**
  Ground truth trajectory from dataset.

* **Intermediate trajectory (τ):**
  Interpolated between τ₀ and τ₁ using time t.

* **Time step (t):**
  Sampled from a uniform distribution between 0 and 1.

* **Context vector (c):**
  Task-specific conditioning information:

  * Tracking → past detections
  * Forecasting → trajectory history
  * Planning → start and goal states

The model input is:
v_θ(t, τ, c)

---

## 2. Model Architecture

### Core Structure

* 1D Temporal U-Net (CNN-based)
* Processes trajectories along time dimension

### Key Components

* **1D Convolutions:**
  Capture temporal dependencies efficiently

* **U-Net Architecture:**
  Encoder-decoder with skip connections

* **FiLM Layers:**
  Inject conditioning information (c)

### Key Idea

The model learns a **time-varying vector field**:
v_θ(t, τ, c)

This defines how to transform noise into realistic trajectories.

---

## 3. Model Output

### Training Output

* Predicted vector field:
  v_θ(t, τ, c)

* Target:
  τ₁ - τ₀

* Loss:
  Mean squared error between predicted and target vector field

---

### Inference Output

* Start from noise τ₀
* Apply ODE-based updates (Euler method)
* Generate final trajectory τ

---

## 4. Key Advantages

* Non-autoregressive → fast generation
* Single-step or few-step sampling possible
* Handles multimodal trajectories
* Much faster than diffusion models (~100×)

---

## 5. Final Output

* Generated trajectory:

  * Future prediction
  * Planned path
  * Multi-modal samples
