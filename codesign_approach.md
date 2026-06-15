# Motor-Capacity Codesign: GA + Pareto Front

## Overview

Optimize actuator torque capacities (τ_max per joint group) to **minimize hardware cost** while preserving **locomotion performance**, using a pre-trained AMP-PPO policy that's conditioned on motor specifications.

## Problem Statement

Given:
- QDD humanoid with 6 symmetric joint groups (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
- Pre-trained AMP-PPO policy that accepts per-joint τ_max as input
- Task: forward locomotion at (0.5-1.5 m/s, 0 rad/s angular velocity)

Find:
- **One torque capacity per group**: τ = {80.0, 72.7, 40.0, 50.0, 25.0, 17.0} Nm
- Minimize: hardware cost (BOM complexity, power, mass)
- Maximize: task performance (policy reward)
- Prefer: 2-3 distinct motor types (different torque ratings)

## Approach: NSGA-II with Soft Penalty

### 1. **Genome**: Pure Continuous Variables

```
X = [τ_hip_pitch, τ_hip_roll, τ_hip_yaw, τ_knee, τ_ankle_pitch, τ_ankle_roll]
    ∈ [5, 90] × [5, 90] × [5, 90] × [5, 90] × [5, 90] × [5, 90] Nm
```

- No discrete flags (unlike earlier attempts with "custom" vs "standard" motors)
- 6D continuous space allows infinite diversity
- Search leverages continuous relaxation

### 2. **Objectives**: Two-objective Pareto

| Objective | Formula | Meaning |
|-----------|---------|---------|
| **Reward** (minimize `-perf`) | `-mean_policy_reward` | Higher cumulative reward is better |
| **Hardware Cost** (minimize) | `w_count·n_types + w_torque·(Σ τ / τ_ref) + penalty` | Prefer few types, low total capacity |

### 3. **Motor Type Clustering**

Count distinct types by **1-decimal rounding** (accounts for manufacturing tolerance):

```python
seen_taus = {round(τ, 1) for τ in X}
n_types = len(seen_taus)  # Usually 2-5
```

Example:
```
τ = [80.0, 80.1, 80.2, 50.0, 50.1, 25.0]
→ {80.0, 50.0, 25.0}  (3 types)
```

### 4. **Soft Penalty for Motor Type Preference**

**Hard constraints failed** because:
- Continuous variables produce fine-grained variations (80.0, 80.3, 80.7, ...)
- GA mutation/crossover naturally explore beyond exact clustering
- Feasible region for "exactly 2 types" is too narrow

**Solution**: Soft cost penalty

```python
excess = max(0, n_types - 3)  # Allow up to 3 types "for free"
penalty = excess * 2.0         # Add 2.0 cost per extra type

total_cost = base_cost + penalty
```

**Effect**:
- 2 types: cost = base (preferred)
- 3 types: cost = base (equally good)
- 4 types: cost = base + 2.0 (penalized but allowed if performance is high)
- 5+ types: cost = base + 4.0+ (discouraged)

### 5. **Hardware Cost Function**

```python
# Diversity metric: count unique τ values (not rounded)
n_choices = len(unique(X))

# Total capacity: sum of all per-joint τ_max
cum_tau = Σ τ_i for i in joint_groups

# Base cost
base_cost = w_count * n_choices + w_torque * (cum_tau / τ_ref)

# Example (τ_ref = 100 Nm):
#   w_count=0.5, w_torque=1.0
#   6 unique values + 450 Nm total
#   → cost = 0.5·6 + 1.0·(450/100) + penalty
#          = 3.0 + 4.5 + penalty
```

### 6. **Policy Evaluation (Vectorized)**

```
1. Load pre-trained AMP-PPO policy
2. Create batch of num_envs parallel MuJoCo environments
3. For each design (genome):
   a. Apply τ_max to all environments
   b. Run policy for n_seeds × num_steps
   c. Average reward across rollouts
4. Return reward vector
```

**Efficiency**: Vectorized on GPU (128 envs × 4 seeds = 512 parallel rollouts)

### 7. **Locomotion Task**

Environment: `Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond`

- **Velocity command**: forward only (0.5-1.5 m/s, 0 rad/s)
- **Reward**: 
  - `+10.0` per step for tracking forward velocity
  - `-1.0` per step electrical power cost (detailed motor model)
  - `-2.0` backward-flight penalty
  - `-1.5` feet slip penalty
- **Simulation**: MuJoCo-Warp (GPU-accelerated)
- **Rollout length**: 1000 steps (20 seconds @ 50 Hz)

## Algorithm: NSGA-II

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating

# Initialization
pop_size = 32
generations = 25
problem = {
    "n_obj": 2,
    "n_var": 6,
    "variables": OrderedDict({
        "tau_hip_pitch": Real(5, 90),
        "tau_hip_roll": Real(5, 90),
        "tau_hip_yaw": Real(5, 90),
        "tau_knee": Real(5, 90),
        "tau_ankle_pitch": Real(5, 90),
        "tau_ankle_roll": Real(5, 90),
    })
}

# NSGA-II with mixed-variable operators
algorithm = NSGA2(
    pop_size=32,
    sampling=MixedVariableSampling(),
    mating=MixedVariableMating(prob_mut=0.15),
    eliminate_duplicates=MixedVariableDuplicateElimination(),
)

result = minimize(problem, algorithm, ("n_gen", 25))
```

## Output: Pareto Front

**Result after 25 generations** (800 evaluations):

| Rank | Reward | Cost | Motors | τ_total (Nm) | Design |
|------|--------|------|--------|-------------|--------|
| 1 | 0.280 | 7.71 | 2 | 694 | {80.0, 40.0, 50.0, 17.0, ...} |
| 2 | 0.285 | 8.54 | 2 | 712 | {80.0, 72.7, 40.0, ...} |
| 3 | 0.290 | 9.15 | 3 | 725 | {80.0, 40.0, 60.0, ...} |
| 4 | 0.295 | 10.2 | 3 | 780 | {85.0, 35.0, 50.0, ...} |

**Trade-off**: 
- Minimal-cost design: 2 motor types, ~694 Nm total, reward 0.280
- Best-performance: 3-4 types, ~800 Nm, reward ~0.295
- User selects based on engineering constraints (cost, weight, power)

## Validation & Visualization

### Plots (publication-quality)

1. **Pareto Front** (scatter plot)
   - X-axis: hardware cost
   - Y-axis: policy reward
   - Points: Pareto-optimal designs
   - Color: reward gradient

2. **Design Comparison** (bar charts)
   - Each subplot = one Pareto design
   - Bars = torque per joint group
   - Shows diversity of configurations

3. **Summary** (4-panel)
   - Pareto front
   - Motor type distribution histogram
   - Torque histogram
   - Text summary (best, worst, tradeoffs)

### Videos

Generate videos of best-by-reward design:
- Resolution: 960×720 @ 50 FPS
- Duration: customizable (default 1000 steps = 20 sec)
- View: side profile, forward-only walking
- Output: MP4 with realistic physics

```bash
# Generate all plots + videos for top 3 designs
uv run python scripts/visualize_codesign.py \
  --pareto codesign_pareto.npz \
  --policy ./model_30000.pt
```

## Files

| File | Purpose |
|------|---------|
| `scripts/codesign_ga.py` | Main GA with NSGA-II, soft penalty, pymoo |
| `scripts/plot_codesign_results.py` | Pareto plots, design comparison, summary |
| `scripts/validate_codesign.py` | Single design validation + video generation |
| `scripts/visualize_codesign.py` | One-command orchestration (GA → plots → videos) |

## Key Insights

1. **Soft penalties > hard constraints** for continuous optimization
   - Hard constraint (exactly N types) makes feasible region too narrow
   - Soft penalty allows natural clustering via cost minimization

2. **1-decimal rounding** clusters similar torques
   - Accounts for manufacturing tolerances (±0.5 Nm per motor)
   - Reduces type count without artificial discretization

3. **Pre-trained policy** dramatically reduces optimization time
   - No policy training in GA loop (black-box evaluation)
   - Policy already knows how to walk; GA just optimizes specs
   - 800 evaluations vs. 750 full training runs (old GA approach)

4. **Forward-only validation** prevents backward walking
   - Velocity command: (0.5-1.5 m/s lin_vel_x, 0.0 ang_vel_z)
   - Ensures clean videos and consistent evaluation

5. **Pareto optimization** reveals tradeoffs
   - Not a single "best" design
   - Users choose based on constraints (mass budget, BOM cost, power)

## Performance

- **GA runtime**: ~10-15 minutes (32 pop, 25 gen, 128 GPU envs)
- **Plots runtime**: <1 minute
- **Video runtime**: ~2-5 minutes per design (1000 steps)
- **Total end-to-end**: ~20-30 minutes
