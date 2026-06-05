# Codesign Reward Formulation

## Overview

The QDD codesign training optimizes **two objectives simultaneously**:

1. **Policy reward** (PPO): trains the locomotion policy
2. **Codesign surrogate loss** (separate optimizer): learns motor-type assignment

These run in an alternating fashion: PPO updates the policy, then the codesign
optimizer updates motor assignments every 10 PPO iterations (after a 2000-iter warmup).

---

## 1. Policy Reward (shapes the walking behavior)

The total per-step reward is a weighted sum of 22 terms. The **codesign-specific
term** is highlighted below.

### Standard locomotion rewards

| Term | Weight | Formula |
|------|--------|---------|
| `track_linear_velocity` | +10.0 | $\exp(-\|v_{xy} - v_{xy}^{\text{cmd}}\|^2 / \sigma^2)$ |
| `track_angular_velocity` | +5.0 | $\exp(-(ω_z - ω_z^{\text{cmd}})^2 / \sigma^2)$ |
| `action_rate_l2` | −0.05 | $\|a_t - a_{t-1}\|^2$ |
| `joint_vel_l1` | −0.01 | $\sum_j |\dot{q}_j|$ |
| `joint_torques_l2` | −1e-9 | $\sum_j \tau_j^2$ |
| `power_consumption` | −1.0 | $\sum_j |\tau_j \cdot \dot{q}_j|$ |
| `no_fly_backward` | −2.0 | Penalize backward flight phase |
| `lin_vel_z_backward` | −2.0 | Penalize vertical velocity |
| `feet_flat_ori` | −1.5 | Penalize non-flat foot orientation |
| `feet_slide` | −1.5 | Penalize foot sliding during contact |
| `standing_double_support` | +2.0 | Reward double-support when standing |
| `dof_pos_limits` | −0.001 | Penalize joints near limits |

*(Other terms like `upright`, `pose`, `air_time`, `foot_clearance`, etc. have weight 0 — they are inactive.)*

### ⭐ Codesign torque penalty (policy-facing)

| Term | Weight | Formula |
|------|--------|---------|
| **`codesign_torque`** | **−0.05** | Composite penalty (see below) |

The composite penalty combines four sub-terms:

$$r_{\text{codesign}} = -0.05 \cdot \left( w_{\text{rms}} \cdot P_{\text{rms}} + w_{\text{peak}} \cdot P_{\text{peak}} + w_{\text{sat}} \cdot P_{\text{sat}} + w_{\text{rate}} \cdot P_{\text{rate}} \right)$$

With current weights: $w_{\text{rms}} = 1.0$, $w_{\text{peak}} = 1.0$, $w_{\text{sat}} = 2.0$, $w_{\text{rate}} = 0.3$.

**Sub-terms:**

1. **RMS torque** — penalizes sustained high effort:
$$P_{\text{rms}} = \frac{\sqrt{\frac{1}{J}\sum_j \tau_j^2}}{\tau_{\text{nom}}}$$

2. **Peak torque** — penalizes instantaneous spikes:
$$P_{\text{peak}} = \frac{\max_j |\tau_j|}{\tau_{\text{nom}}}$$

3. **Saturation proximity** — penalizes operating near effort limits (uses learned $\tau_{\text{max}}^j$):
$$P_{\text{sat}} = \frac{1}{J}\sum_j \sigma\left(15 \cdot \left(\frac{|\tau_j|}{\tau_{\text{max}}^j} - 0.85\right)\right)$$

   > **This is where codesign enters the policy reward**: the effort limits $\tau_{\text{max}}^j$ are set by the codesign module and change during training.

4. **Torque rate** — penalizes sudden torque changes:
$$P_{\text{rate}} = \frac{\sum_j (\tau_j^t - \tau_j^{t-1})^2}{\tau_{\text{nom}}^2}$$

Where $\tau_{\text{nom}} = 80$ Nm (reference normalization), $J = 12$ joints.

---

## 2. Codesign Surrogate Loss (optimizes motor assignment)

This is a **separate loss** optimized by a dedicated Adam optimizer (lr=3e-3)
every 10 PPO iterations. It operates on logged torques from the policy rollout.

### Learnable parameters

- **$\alpha \in \mathbb{R}^{U \times K}$**: Assignment logits ($U$ = 6 unique joints due to symmetry, $K$ = 3 motor types)
- **$\tau_{\text{raw}} \in \mathbb{R}^K$**: Unbounded params mapped to $\tau_{\text{max}}$ via bounded sigmoid: $\tau_{\text{max}}^k = \tau_{\min} + (\tau_{\max} - \tau_{\min}) \cdot \sigma(\tau_{\text{raw}}^k)$

### Effective torque limit per joint

Using Gumbel-Softmax for differentiable discrete selection:

$$\tau_{\text{eff}}^j = \sum_k p_{jk} \cdot \tau_{\text{max}}^k$$

where $p_{jk} = \text{GumbelSoftmax}(\alpha_j / T)$ during training, or $\text{argmax}$ at eval.

### Loss components

$$\mathcal{L} = \underbrace{\lambda_{\text{rms}} \cdot \frac{\text{RMS}(\hat{\tau})}{\tau_{\text{ref}}}}_{\text{RMS penalty}} + \underbrace{\lambda_{\text{peak}} \cdot \frac{\max|\hat{\tau}|}{\tau_{\text{ref}}}}_{\text{Peak penalty}} + \underbrace{\lambda_{\text{sat}} \cdot \sigma(20 \cdot (|\hat{\tau}|/\tau_{\text{eff}} - 0.8))}_{\text{Saturation risk}} + \underbrace{\mathcal{L}_{\text{struct}}}_{\text{Structural}}$$

Where $\hat{\tau} = \tau_{\text{eff}} \cdot \tanh(\tau_{\text{logged}} / \tau_{\text{eff}})$ (differentiable soft clipping).

**Current lambda values:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lambda_rms` | 0.01 | Minimize average torque usage |
| `lambda_peak` | 0.02 | Minimize peak torque |
| `lambda_saturation` | 0.05 | Avoid operating near limits |
| `lambda_tau` | 1e-4 | Prefer smaller motors |
| `lambda_types` | 0.01 | Prefer fewer distinct types |
| `lambda_balance` | 0.05 | Prevent one type dominating |

**Structural loss:**

$$\mathcal{L}_{\text{struct}} = \lambda_{\text{types}} \cdot \sum_k \sigma(10 \cdot (\bar{p}_k - 0.1)) + \lambda_{\text{balance}} \cdot \text{ReLU}(\max_k \bar{p}_k - 0.7)^2 + \lambda_{\text{tau}} \cdot \sum_k \tau_{\text{max}}^k$$

---

## 3. Interaction between policy and codesign

```
┌─────────────────────────────────────────────────────┐
│              Training Loop (alternating)              │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─── PPO (every iter) ───┐                           │
│  │ Rollout with current    │                           │
│  │ effort limits τ_max^j   │──→ torque logs            │
│  │ Reward includes P_sat   │                           │
│  │ which uses τ_max^j      │                           │
│  └─────────────────────────┘                           │
│              ↑                      ↓                  │
│     τ_max^j updated          logged torques            │
│              ↑                      ↓                  │
│  ┌─── Codesign (every 10 iters) ───┐                  │
│  │ Surrogate loss on logged τ      │                   │
│  │ Update α (assignment) and       │                   │
│  │ τ_raw (motor capacities)        │                   │
│  │ Set new effort limits           │                   │
│  └──────────────────────────────────┘                  │
│                                                       │
└─────────────────────────────────────────────────────┘
```

**Key insight**: The policy "sees" the codesign decision through the saturation
proximity penalty. As the codesign module tightens limits on certain joints,
the policy learns to use less torque on those joints. Conversely, if the policy
needs high torque on a joint, the codesign module is encouraged to assign a
larger motor type to it.
