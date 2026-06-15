# Motor Codesign: Overview & Quick Reference

**For detailed comparison of both approaches, see [CODESIGN_APPROACHES.md](CODESIGN_APPROACHES.md).**

This document provides a quick reference for the **current working implementation** (NSGA-II).

---

## Quick Start

```bash
# Run GA (10-15 min on 128 GPU envs)
uv run python scripts/codesign_ga.py \
  --backend mjlab \
  --task Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond \
  --policy ./model_30000.pt \
  --pop-size 32 --generations 25

# Generate plots + best design video (5-10 min)
uv run python scripts/visualize_codesign.py \
  --pareto codesign_pareto.npz \
  --policy ./model_30000.pt
```

---

## Current Approach: NSGA-II with Soft Penalty

### Problem

Optimize **torque capacities** (τ_max per joint group) to minimize hardware cost while preserving locomotion performance. Prefer 2-3 distinct motor types.

### Solution

Multi-objective genetic algorithm with:
- **6D continuous search space**: one τ per joint group
- **Two objectives**: reward (maximize) vs cost (minimize)
- **Soft penalty**: prefer ≤3 motor types via cost function
- **Pre-trained policy**: frozen AMP-PPO (black-box evaluation)
- **GPU-accelerated**: 128 parallel envs × 4 seeds = 512 rollouts/gen

### Genome

```python
X = [τ_hip_pitch, τ_hip_roll, τ_hip_yaw, τ_knee, τ_ankle_pitch, τ_ankle_roll]
    ∈ [5, 90]^6 Nm
```

### Objectives

```python
f_perf = -mean_policy_reward              # higher reward = better
f_cost = w_count·n_types + w_torque·(Στ/τ_ref) + penalty

penalty = max(0, n_types - 3) * 2.0       # soft preference for ≤3 types
```

### Motor Type Clustering

1-decimal rounding (manufacturing tolerance):
```python
unique_types = {round(τ, 1) for τ in X}
n_types = len(unique_types)
```

### Output: Pareto Front

Non-dominated designs revealing performance-cost tradeoff:

| Reward | Cost | Motors | τ_total (Nm) | Design |
|--------|------|--------|-------------|--------|
| 0.280 | 7.71 | 2 | 694 | {80.0, 40.0, 50.0, ...} |
| 0.285 | 8.54 | 2 | 712 | {80.0, 72.7, ...} |
| 0.290 | 9.15 | 3 | 725 | {80.0, 40.0, 60.0, ...} |
| 0.295 | 10.2 | 3 | 780 | {85.0, 35.0, 50.0, ...} |

User selects based on engineering constraints (mass budget, BOM cost, power).

---

## Why This Approach Works

1. **Soft penalty > hard constraint**
   - Hard constraints (exactly N types) too restrictive
   - Soft penalty allows natural clustering via cost minimization

2. **1-decimal rounding**
   - Accounts for manufacturing tolerance (±0.5 Nm/motor)
   - Clusters similar torques without artificial discretization

3. **Pre-trained policy**
   - No policy training in GA loop (black-box evaluation)
   - 800 evaluations ≈ 10-15 min (vs. 750+ full training runs)

4. **Forward-only walking**
   - Velocity command: (0.5-1.5 m/s forward, 0 rad/s angular)
   - Ensures clean videos and consistent evaluation

5. **Pareto optimization**
   - Not a single "best" design
   - All tradeoffs visible
   - Users choose based on constraints

---

## Validation & Visualization

### Plots
- **Pareto front**: scatter plot (cost vs reward)
- **Design comparison**: bar charts (τ per joint group)
- **Summary**: 4-panel overview with histograms and metrics

### Videos
- Resolution: 960×720 @ 50 FPS
- Duration: customizable (default 1000 steps = 20 sec)
- Forward-only walking with realistic physics

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/codesign_ga.py` | NSGA-II GA with pymoo, soft penalty |
| `scripts/plot_codesign_results.py` | Pareto plots, design comparison |
| `scripts/validate_codesign.py` | Single design validation + video |
| `scripts/visualize_codesign.py` | One-command orchestration |
| `CODESIGN_APPROACHES.md` | **Full comparison: Gumbel-Softmax vs NSGA-II** |

---

## Performance

- **GA runtime**: ~10-15 min (32 pop, 25 gen, 128 GPU envs)
- **Plots runtime**: <1 min
- **Video runtime**: ~2-5 min per design (1000 steps)
- **Total end-to-end**: ~20-30 min

---

## See Also

- [CODESIGN_APPROACHES.md](CODESIGN_APPROACHES.md) — Detailed comparison of **both approaches** (Gumbel-Softmax & NSGA-II)
- [README.md](README.md) — Project overview
