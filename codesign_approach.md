# Motor Codesign: Current Architecture (Clustering-Based NSGA-II)

**For full comparison with differentiable Gumbel-Softmax, see
[CODESIGN_APPROACHES.md](CODESIGN_APPROACHES.md).**

This document describes the current production workflow used by
`scripts/codesign_ga.py` and `scripts/plot_codesign_results.py`.

---

## Quick Start

```bash
# 1) Run co-design GA
uv run python scripts/codesign_ga.py \
  --backend mjlab \
  --task Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond \
  --policy ./model_30000.pt \
  --pop-size 32 --n-seeds 4 --generations 25

# 2) Generate visual analysis (includes clustering plot)
uv run python scripts/plot_codesign_results.py \
  --pareto codesign_pareto.npz \
  --output-dir codesign_results

# 3) Optional: validate/video selected design
uv run python scripts/visualize_codesign.py \
  --pareto codesign_pareto.npz \
  --policy ./model_30000.pt \
  --criteria efficiency
```

---

## Architecture Summary

### 1) Design space (continuous torques)

The genome is 6D continuous, one `tau_*` per symmetric joint group:

- `tau_hip_pitch` in `[30, 150]`
- `tau_hip_roll` in `[30, 150]`
- `tau_hip_yaw` in `[10, 100]`
- `tau_knee` in `[30, 180]`
- `tau_ankle_pitch` in `[15, 120]`
- `tau_ankle_roll` in `[5, 80]`

### 2) Evaluation backend

- Frozen motor-conditioned policy (`model_30000.pt`)
- MuJoCo-Warp physics via mjlab backend
- Per-design rollout reward is measured directly (no policy retraining in loop)

### 3) Multi-objective optimization

NSGA-II minimizes:

```python
f_perf = -mean_reward
f_cost = w_count * n_choices + w_torque * (cum_tau / tau_ref)
         + max(0, n_types - 3) * w_motor_type_penalty
```

Current defaults in `codesign_ga.py`:

- `w_count = 0.5`
- `w_torque = 1.0`
- `tau_ref = 90.0`
- `w_motor_type_penalty = 50.0`

### 4) Objective sign convention (important)

The first objective stored in Pareto files is **negative reward**:

```python
F[:, 0] = -reward
```

Therefore:

- best-by-reward index = `argmin(F[:, 0])`
- displayed reward = `-F[i, 0]`
- default final-selection criterion for visualization/validation = `efficiency`
  (maximize `reward / cost`)
- efficiency selection is walkability-aware: it first filters to designs with
  reward >= `0.97 * max_reward`, then picks max `reward / cost` in that set

All plotting and reporting should use this convention to avoid selecting the
worst-performing design by mistake.

---

## Motor-Type Counting with Clustering

The old approach that treated tiny torque differences as distinct "types" was
replaced by hierarchical clustering.

### Current rule

- Method: complete-linkage hierarchical clustering
- Distance: Euclidean on torque values
- Cut threshold: `TORQUE_CLUSTER_THRESHOLD_NM = 12.0`

This threshold is intentionally wider, so values like `25, 31, 36 Nm` are
treated as one cluster (same motor type family), not three fake types.

### Why this change

With a 2 Nm threshold, near-identical torques were split into multiple types,
which made best-design reports look inconsistent and over-fragmented.

---

## Visualization Pipeline (with clustering)

`plot_codesign_results.py` now produces:

- `pareto_front.png`
- `design_comparison.png`
- `codesign_summary.png`
- `motor_clustering.png` (**new**)

### Plot semantics

- Joint bars/heatmap are colored by **torque clusters**
- Text overlays show only torque values (Nm), not `T1/T2/T3`
- `motor_clustering.png` shows dendrogram + threshold cut line for the best
  design, making cluster grouping explicit

---

## Interpreting "best design"

The Pareto "best by reward" can have low and close torques. This is valid if:

1. Reward objective favors that region.
2. Cost objective and clustering penalty do not force larger separations.
3. Cluster threshold merges close values into one motor type class.

If you see close torques reported as multiple types, check the clustering
threshold and rerun plotting with current scripts.

If you see a "best" design that does not walk, first verify the selection
convention above (`argmin(F[:, 0])`) before changing GA hyperparameters.

---

## Relevant Files

| File | Role |
|---|---|
| `scripts/codesign_ga.py` | NSGA-II optimization and clustered motor-type counting |
| `scripts/plot_codesign_results.py` | Cluster-aware plots + dendrogram (`motor_clustering.png`) |
| `scripts/validate_codesign.py` | Rollout validation and video generation |
| `scripts/visualize_codesign.py` | One-command plotting + validation pipeline |
| `CODESIGN_APPROACHES.md` | NSGA-II vs Gumbel-Softmax comparison |

---

## Notes

- Gumbel-Softmax codesign remains in the repository for parallel investigation.
- This document only describes the current GA workflow used for batch design
  search and Pareto analysis.
- Keep this file updated whenever co-design selection logic, clustering
  thresholds, or output plots change.
