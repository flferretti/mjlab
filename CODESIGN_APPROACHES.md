# Motor Codesign: Comparing Two Approaches

This document compares two distinct methods for optimizing actuator specifications alongside locomotion policy training/evaluation.

---

## Problem Statement (Both Approaches)

Given a QDD humanoid with N=6 symmetric joint groups:
- Find optimal **torque capacities** (τ_max per joint group)
- Minimize **hardware cost** (BOM complexity, power, mass)
- Maximize **locomotion performance** (forward walking)
- Prefer **2-3 distinct motor types** (reduce supply chain complexity)

---

# Approach 1: Gumbel-Softmax Co-Training

## Overview

Make motor assignment **differentiable** and optimize it **during policy training** using soft distributions that harden over time.

### Key Concept: Soft Motor Assignment

Each joint gets a **learned soft distribution** over K candidate motor types:

```
p[joint, k] = softmax(α[joint, k])      # learnable logits
τ_eff[joint] = Σ_k p[joint, k] × τ_max[k]  # effective torque limit
```

Early training: soft (all types contribute equally)  
Late training: hard (one type per joint via temperature → 0)

### Architecture

```
Policy Gradient (PPO)
    ↓
    ├→ Policy: π(a | obs, τ_eff)    [learn to walk with current motors]
    └→ Log torques: τ_applied
         ↓
    Codesign Optimizer (Adam)
         ↓
    ├→ Surrogate Loss (on logged τ_applied)
    │   - RMS torque penalty
    │   - Peak torque penalty  
    │   - Saturation risk (>85% limit)
    │   - Type count penalty
    │   - Balance penalty
    │   - τ_max pressure (prefer smaller)
    └→ Update: α, τ_raw
         ↓
    Back to Policy Gradient (next iteration)
```

### Temperature Annealing

```python
temperature = T_init × (T_final / T_init) ** (iteration / n_iterations)
# Typical: T_init=2.0, T_final=0.1, n_iterations=100k
```

Gumbel-Softmax at iteration i:
```
sampled = (log(p) + gumbel_noise) / temperature
hard_assignment = argmax(sampled)
```

Higher T → samples are soft (diffuse probabilities)  
Lower T → samples are hard (one-hot-ish)

### Symmetry Enforcement

Left-right joint pairs share assignment logits:
```python
# l_hip_pitch and r_hip_pitch share α
p_l_hip = p_r_hip = softmax(α_hip_pitch)
τ_l_hip = τ_r_hip = τ_eff_hip_pitch
```

### Warmup Phase

- **Iterations 0-2000**: Policy learns to walk without motor optimization
- **Iterations 2000+**: Codesign optimizer activates, begins tuning motors
- **Torque rewards**: Active from start, encouraging efficiency

### Output at Convergence

Extract final discrete assignment:
```python
hard_assignment[joint] = argmax(p[joint, :])      # which type
hard_tau_max[type] = exp(tau_raw[type])           # the torque limit
```

### Pros & Cons

| Pros | Cons |
|------|------|
| Single training run | Complex gradient flow (non-differentiable sim) |
| Co-evolved design + policy | Temperature tuning is critical |
| Can find novel motor specs | Surrogate loss heuristic; may not match true torque profile |
| Data-efficient | Harder to debug (many hyperparameters) |
| | Requires careful warmup |

---

# Approach 2: NSGA-II with Pre-Trained Policy

## Overview

Use a **frozen, pre-trained policy** as black-box evaluator and search for optimal motor specs via multi-objective genetic algorithm.

### Key Concept: Pareto Optimization

Two competing objectives:
1. **Maximize reward** (policy performance with these motors)
2. **Minimize cost** (hardware cost: motor count + total capacity)

The **Pareto front** shows all non-dominated tradeoffs.

### Genome: Pure Continuous Variables

```python
X = [τ_hip_pitch, τ_hip_roll, τ_hip_yaw, τ_knee, τ_ankle_pitch, τ_ankle_roll]
    ∈ [5, 90] × [5, 90] × [5, 90] × [5, 90] × [5, 90] × [5, 90] Nm
```

No discrete flags; 6D continuous space allows infinite diversity.

### Motor Type Clustering via 1-Decimal Rounding

```python
seen_taus = {round(τ, 1) for τ in X}
n_types = len(seen_taus)
```

Example:
```
X = [80.0, 80.1, 80.2, 50.0, 50.1, 25.0]
→ {80.0, 50.0, 25.0}  (3 types, after rounding)
```

Accounts for manufacturing tolerance (±0.5 Nm/motor).

### Hardware Cost Function

```python
n_choices = len(unique(X))              # diversity metric
cum_tau = Σ X                            # total capacity
excess = max(0, n_types - 3)            # motor types above 3
penalty = excess * 2.0                  # soft penalty per excess type

base_cost = w_count * n_choices + w_torque * (cum_tau / τ_ref)
total_cost = base_cost + penalty
```

Effect:
- 2-3 types: cost = base (preferred)
- 4+ types: cost = base + 2.0 × (excess) (penalized but allowed)

**Why soft penalty instead of hard constraint?**
- Hard constraint (exactly N types) creates infeasibly narrow feasible region
- Continuous mutation produces fine-grained variations (80.0, 80.3, 80.7, ...)
- Soft penalty lets GA naturally cluster via cost minimization

### Objectives: NSGA-II with pymoo

```python
F = [f_perf, f_cost]
f_perf = -mean_policy_reward        # minimize negative = maximize reward
f_cost = hardware cost function     # minimize cost

# NSGA-II balances both via Pareto dominance
# Pareto front: all designs where no other improves both objectives
```

### Evaluation Loop

```
For each design X in population:
  1. Apply τ_max = X to environment
  2. Run pre-trained policy for n_seeds × n_steps
  3. Average reward across rollouts
  4. Compute cost from X
  5. Return (reward, cost)
```

Vectorized on GPU: 128 parallel envs × 4 seeds = 512 simultaneous rollouts.

### Algorithm: NSGA-II with Mixed-Variable Operators

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating

algorithm = NSGA2(
    pop_size=32,
    sampling=MixedVariableSampling(),      # handles dict-based continuous vars
    mating=MixedVariableMating(),          # crossover + mutation
    eliminate_duplicates=MixedVariableDuplicateElimination(),
)

result = minimize(problem, algorithm, ("n_gen", 25))
# 25 generations × 32 pop = 800 evaluations ≈ 10-15 min on 128 GPU envs
```

### Pareto Front Output

After 25 generations:

| Rank | Reward | Cost | Motors | τ_total (Nm) | Design |
|------|--------|------|--------|-------------|--------|
| 1 | 0.280 | 7.71 | 2 | 694 | {80.0, 40.0, 50.0, 17.0, ...} |
| 2 | 0.285 | 8.54 | 2 | 712 | {80.0, 72.7, 40.0, ...} |
| 3 | 0.290 | 9.15 | 3 | 725 | {80.0, 40.0, 60.0, ...} |
| 4 | 0.295 | 10.2 | 3 | 780 | {85.0, 35.0, 50.0, ...} |

**User selects based on constraints**: cost budget? performance target? weight limit?

### Pros & Cons

| Pros | Cons |
|------|------|
| Simple, interpretable algorithm | Requires pre-trained policy (can't co-optimize) |
| No gradient required (black-box) | More evaluations than Gumbel-Softmax |
| Clear Pareto tradeoffs | Lower design optimality (policy is frozen) |
| Easy to parallelize on GPU | Can't discover novel gaits |
| Robust to noisy fitness | |

---

# Detailed Comparison

## Conceptual Differences

| Aspect | Gumbel-Softmax | NSGA-II |
|--------|---|---|
| **Search strategy** | Gradient-based co-optimization | Population-based evolutionary search |
| **Policy** | Trains from scratch, co-adapts | Pre-trained, frozen |
| **Motor assignment** | Soft → hard via temperature | Direct continuous optimization |
| **Gradients** | Surrogate loss on logged torques | None (black-box evaluation) |
| **Iterations** | ~100k PPO steps | ~800 policy evaluations |
| **Time** | Depends on training (1-2 hours?) | ~10-15 min (vectorized) |

## Optimization Landscape

### Gumbel-Softmax
```
Search space: [α logits, τ_raw values]  (continuous, smooth)
Objective: PPO reward (noisy, from env) + surrogate loss
Navigation: Gradient descent with annealing
Convergence: Should smooth; temperature annealing helps escape local minima
```

### NSGA-II
```
Search space: [τ per joint]  (6D continuous, clean)
Objective: Policy reward (from frozen policy) + hardware cost
Navigation: Mutation, crossover, selection
Convergence: Population maintains diversity; Pareto front stabilizes over gens
```

## Hardware Cost Modeling

### Gumbel-Softmax
- **Surrogate loss** penalizes logged torques
- Encourages low RMS, low peak, low saturation
- **Does not** directly minimize hardware cost
- May pick motors that are torque-efficient but expensive (rare types)

### NSGA-II
- **Direct hardware cost function**:
  - w_count × n_types (favor fewer motor models)
  - w_torque × total_capacity (favor lower torque specs)
  - soft_penalty × excess_types (prefer 2-3 types)
- Explicitly models BOM complexity

## Policy Co-Adaptation

### Gumbel-Softmax
- ✅ Policy learns with motors
- ✅ Can discover novel motor specs optimized for learned gait
- ❌ Complex: must tune temperature, warmup, surrogate loss weights
- ❌ Risk of local minima (policy stuck, motors not useful)

### NSGA-II
- ✅ Simple: just evaluate with frozen policy
- ✅ Robust: policy doesn't drift
- ❌ Limited by pre-trained policy capabilities
- ❌ Can't discover gaits that require novel motor specs

---

# Validation & Visualization

Both approaches use the same validation pipeline:

### Locomotion Task

- Environment: `Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond`
- Command: Forward-only (0.5-1.5 m/s, 0 rad/s angular)
- Reward: +10.0 velocity tracking, -1.0 power, -1.5 feet slip, -2.0 backward
- Physics: MuJoCo-Warp (GPU-accelerated)

### Videos

```bash
# For Gumbel-Softmax: validate final hard_assignment
uv run python scripts/validate_codesign.py \
  --policy ./final_model.pt \
  --design-idx 0 \
  --rollout-steps 1000 \
  --output-video gumbel_walk.mp4

# For NSGA-II: validate best-by-reward design
uv run python scripts/validate_codesign.py \
  --policy ./model_30000.pt \
  --pareto codesign_pareto.npz \
  --design-idx 0 \
  --rollout-steps 1000 \
  --output-video ga_walk.mp4
```

### Plots

**Gumbel-Softmax:**
- Training curves: reward, codesign loss, temperature
- Final assignment histograms (which motor per joint)
- Final τ_max per type
- Torque profiles during rollout

**NSGA-II:**
- Pareto front (scatter: cost vs reward)
- Design comparison (bar charts: τ per group)
- Motor type distribution
- Generation-by-generation convergence

---

# Recommended Usage

## Use Gumbel-Softmax if:
- You want to **discover novel motor specs** optimized for learned behavior
- You have **plenty of GPU time** for training
- You're willing to **tune hyperparameters** (temperature, warmup, surrogate loss)
- You want **a single final design** (not multiple tradeoffs)
- You expect **co-adaptation of policy + motors** is crucial

## Use NSGA-II if:
- You have a **robust pre-trained policy** you trust
- You want **fast results** (10-15 min vs 1-2 hours)
- You want to **see all tradeoffs** (Pareto front)
- You prefer **simple, interpretable** optimization
- You want to **compare designs** empirically (videos, plots)

## Best Practice: Run Both

1. **Run NSGA-II first** (10-15 min)
   - Get a Pareto front quickly
   - Visualize reward-cost tradeoffs
   - Select 2-3 candidate designs

2. **For selected designs, optionally run Gumbel-Softmax**
   - Use best-by-reward design as initialization
   - Fine-tune for 10-20k additional PPO steps
   - See if motors + policy co-adapt further

3. **Compare final results**
   - NSGA-II: robust, diverse, fast
   - Gumbel-Softmax: potentially novel, but needs validation

---

# Files & Scripts

## Current Implementation Status

### Gumbel-Softmax (Historical, May Need Revival)
- Original: `src/mjlab/rl/amp_runner.py` (CodesignAMPOnPolicyRunner)
- Configs: `src/mjlab/tasks/velocity/config/qdd/codesign_env_cfg.py`
- Status: ⚠️ Implemented but may need testing/debugging after recent changes

### NSGA-II (Fully Tested & Working)
- GA Script: `scripts/codesign_ga.py` ✅
- Plotting: `scripts/plot_codesign_results.py` ✅
- Validation: `scripts/validate_codesign.py` ✅
- Orchestration: `scripts/visualize_codesign.py` ✅

---

# Next Steps

To investigate both approaches:

1. **Test NSGA-II** (already working):
   ```bash
   uv run python scripts/codesign_ga.py \
     --backend mjlab \
     --task Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond \
     --policy ./model_30000.pt \
     --pop-size 32 --generations 25
   
   uv run python scripts/visualize_codesign.py \
     --pareto codesign_pareto.npz \
     --policy ./model_30000.pt
   ```

2. **Revive & test Gumbel-Softmax** (historical implementation):
   - Check `CodesignAMPOnPolicyRunner` in amp_runner.py
   - Verify temperature annealing, surrogate loss
   - Test on same task with same policy initialization
   - Compare results

3. **Collect metrics for both**:
   - Convergence speed
   - Final design diversity (motor types)
   - Locomotion quality (videos, reward)
   - Computational cost

4. **Write comparison paper/report** with findings

---

# References

- Gumbel-Softmax: Jang et al. "Categorical Reparameterization with Gumbel-Softmax" (2016)
- NSGA-II: Deb et al. "A Fast and Elitist Multiobjective Genetic Algorithm" (2002)
- pymoo: Blank & Deb "Pymoo: Multi-Objective Optimization in Python" (2020)
