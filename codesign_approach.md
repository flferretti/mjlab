# Motor-Type Codesign: Gumbel-Softmax vs GA

## The Problem

Given a robot with N actuated joints, find the minimum number of motor types and assign each joint to a type, while simultaneously training a locomotion policy.

## GA Approach (Previous)

The Genetic Algorithm treats motor selection as a black-box combinatorial problem:

1. **Genome** = discrete motor assignment per joint (e.g., `[AJD12, AJD10, AJD8, ...]`)
2. **Fitness** = train a full RL policy per genome, evaluate performance
3. **Search** = tournament selection, crossover, mutation across a population

**Cost**: `population_size × generations × full_RL_training` — each fitness evaluation requires training a policy from scratch. For 50 genomes × 15 generations, that's 750 training runs.

## Gumbel-Softmax Approach (Current)

Instead of searching over discrete assignments externally, we make the assignment **differentiable** and optimize it **inside** a single training run.

### Key Idea

Each joint doesn't get a hard motor type — it gets a **soft probability distribution** over K candidate types. The motor's effective torque limit becomes a weighted average:

```
τ_eff[joint] = Σ_k  p[joint, k] × τ_max[k]
```

where `p[joint, k]` are softmax probabilities (learnable logits `α`), and `τ_max[k]` are the per-type torque limits (also learnable, sigmoid-bounded).

### Gumbel-Softmax Trick

To make discrete selection differentiable, we use Gumbel-Softmax with temperature annealing:
- **High temperature** (early training): soft assignments, all types contribute → exploration
- **Low temperature** (late training): assignments sharpen toward one-hot → converges to discrete selection

### Alternating Optimization

PPO rewards don't provide gradients to motor parameters (the physics sim isn't differentiable). So we alternate:

1. **PPO step**: train the policy with current motor limits (standard RL)
2. **Codesign step**: optimize motor parameters using a **surrogate loss** on logged torques

The surrogate loss re-applies soft saturation differentiably to the torques the policy actually produced, and backpropagates through `α` (assignment logits) and `τ_raw` (type capacities).

### Surrogate Loss Components

The codesign optimizer minimizes:

| Component | What it does |
|-----------|-------------|
| **RMS torque** | Penalizes sustained high torque (thermal stress) |
| **Peak torque** | Penalizes worst-case spikes |
| **Saturation risk** | Soft step penalty when operating >85% of limit |
| **Type count** | Penalizes having many distinct active types |
| **Balance** | Prevents outlier assignments (1 joint S, 11 joints M) |
| **τ pressure** | Downward pressure on τ_max — prefer smaller motors |

### Symmetry Enforcement

Left-right joint pairs (e.g., `l_hip_pitch` / `r_hip_pitch`) share assignment logits — they always get the same motor type. This halves the search space and ensures a physically symmetric robot.

### Warmup

The codesign module doesn't modify effort limits for the first N iterations (default: 2000). This lets the policy learn to walk before motor optimization begins. Torque penalty rewards are active from the start, teaching the policy to be torque-efficient.

## Comparison

| | GA | Gumbel-Softmax |
|---|---|---|
| **Cost** | O(pop × gen × training) | O(1 training) |
| **Gradient signal** | None (black-box) | Surrogate loss on logged torques |
| **Motor params** | Fixed from catalog | Learnable (τ_max optimized continuously) |
| **Assignment** | Hard discrete | Soft → hard via temperature annealing |
| **Policy interaction** | Independent per genome | Co-evolves with policy |

## Output

At convergence, extract:
- `hard_assignment()`: which motor type each joint gets
- `hard_tau_max()`: the optimized τ_max per type (Nm)
- These define the minimum motor spec needed for each joint group
