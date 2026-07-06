# Motor Codesign: Current Architecture (Clustering-Based NSGA-II)

**For full comparison with differentiable Gumbel-Softmax, see
[CODESIGN_APPROACHES.md](CODESIGN_APPROACHES.md).**

This document describes the current production workflow used by
`scripts/codesign_ga.py` and `scripts/plot_codesign_results.py`.

---

## Quick Start

```bash
# 1a) Recommended: true bi-objective NSGA-II (whole front in one run) with the
#     grounded power-law mass model.
uv run python scripts/codesign_ga.py \
  --backend mjlab \
  --task Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond \
  --policy ./model_30000.pt \
  --pop-size 32 --n-seeds 4 --generations 25 \
  --multiobjective --max-motor-types 3 \
  --cost-model powerlaw --cost-alpha 0.75

# 1b) Legacy: scalarized single-objective, sweep w_torque to trace the front.
uv run python scripts/codesign_ga.py \
  --backend mjlab --task Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond \
  --policy ./model_30000.pt --pop-size 32 --n-seeds 4 --generations 25 \
  --w-torque-sweep 0.25,0.5,1,2,4,8

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

For the **Gene01** whole-body humanoid use `--robot gene` (Isaac Lab backend), and
`scripts/validate_codesign_gene.py` to select + validate + plot the chosen design.

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

### 3) Optimization modes

`codesign_ga.py` supports two optimization modes:

**(a) Scalarized single-objective + `w_torque` sweep (default).**
The pymoo problem is `n_obj=1`, minimizing a single scalarized fitness:

```python
f_perf = -mean_reward
f_cost = w_count * n_choices
       + w_torque * capacity_cost(tau)          # mass proxy, see below
       + max(0, n_types - 3) * w_motor_type_penalty
fitness = f_perf + f_cost                        # minimized
```

A single run yields **one** design (the scalar optimum for that weight). To trace a
reward-vs-cost Pareto front you sweep `w_torque` (`--w-torque-sweep 0.25,0.5,1,2,4,8`),
running the GA once per value and aggregating the winners. This is simple but each
front point is a separate stochastic run, so the front is noisy.

**(b) True bi-objective NSGA-II (`--multiobjective`, recommended).**
The problem is `n_obj=2`, minimizing `(-reward, actuator_mass)` directly, with the
motor-diversity limit enforced as a hard inequality constraint
(`n_types <= --max-motor-types`). NSGA-II returns the **whole non-dominated front in
one run** — no `w_torque` scalarization or sweep, no weight tuning, and a smoother
front. This is the preferred mode; the `w_torque` sweep is kept for backwards
compatibility and A/B comparison.

Current scalarization defaults in `codesign_ga.py`:

- `w_count = 1.0`
- `w_torque = 1.0`
- `tau_ref = 90.0` (legacy proxy only)
- `w_motor_type_penalty = 50.0`

### 3b) Actuator mass / cost models (`codesign_motor_model.py`)

The hardware-capacity term treats a motor's **peak torque as a proxy for its
mass/size**. The original proxy was *linear* (`cum_tau = sum(tau)`, i.e. mass ∝ τ),
which is physically wrong: for BLDC / quasi-direct-drive actuators mass scales
**sub-linearly** with peak torque,

```
m(τ) ≈ k · τ^α ,   α ≈ 0.7–0.8   (torque density slowly improves with size)
```

`--cost-model` selects the capacity term (`capacity_cost` in `CodesignConfig`):

| model | capacity term | when to use |
|---|---|---|
| `legacy` | `cum_tau / tau_ref` (linear, dimensionless) | reproduce original behavior |
| `linear` | total mass, `m ∝ τ` (kg) | linear baseline, anchored to 1.2 kg @ 100 Nm |
| `powerlaw` | `m = k·τ^α`, `α=--cost-alpha` (kg) | **grounded default** for BLDC/QDD |
| `catalog` | nearest real SKU with `τ_peak ≥ τ` (kg) | procurement: exact BoM + SKU count |

The `catalog` model snaps each joint to the cheapest SKU that covers its torque
demand, so it yields an exact mass, a bill-of-materials, and a *natural* distinct-SKU
count (no post-hoc clustering needed). The default catalog (`DEFAULT_CATALOG`) is a
representative humanoid-QDD line (40–200 Nm) — replace with real datasheet values
when a vendor is chosen. Masses are anchored so a 100 Nm actuator ≈ 1.2 kg, keeping
the capacity term of comparable magnitude to the legacy proxy (so `w_torque` stays
interpretable). `linear` and `powerlaw` agree at the 100 Nm anchor; below it the
power law charges *more* per Nm (small motors have worse torque density), above it
*less*.

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
| `scripts/codesign_ga.py` | GA optimization (scalarized sweep + bi-objective NSGA-II) and clustered motor-type counting |
| `scripts/codesign_motor_model.py` | Actuator mass/cost models (linear / power-law / catalog) + motor catalog |
| `scripts/test_codesign_motor_model.py` | Unit tests for the mass/cost models |
| `scripts/plot_codesign_results.py` | Cluster-aware plots + dendrogram (`motor_clustering.png`) |
| `scripts/validate_codesign.py` | QDD rollout validation and video generation |
| `scripts/validate_codesign_gene.py` | Gene01 design selection + Isaac validation + Pareto plot |
| `CODESIGN_APPROACHES.md` | NSGA-II vs Gumbel-Softmax comparison |

---

## Research notes & improvement roadmap

The capacity cost is only as good as its **torque → mass/size** model. Ordered by
expected impact / effort:

1. **Sub-linear mass scaling (done).** `m ∝ τ^α`, α≈0.75 for BLDC/QDD
   (`--cost-model powerlaw`). Replaces the physically-wrong linear `cum_tau`.
2. **True bi-objective front (done).** `--multiobjective` NSGA-II returns the whole
   reward-vs-mass front in one run; removes the noisy `w_torque` sweep.
3. **Discrete catalog (done, needs real data).** `--cost-model catalog` snaps to real
   SKUs → exact mass, bill-of-materials, natural SKU count. Populate `DEFAULT_CATALOG`
   with datasheet `(τ_peak, τ_cont, ω_max, mass, price)` rows.
4. **Size on the demanded envelope, not the search bound (todo).** Log the actual
   `(τ, ω)` demanded during the rollout and size each motor by the *thermal RMS*
   torque and *peak power* it must deliver, not just the design's peak-torque bound.
   Peak torque alone under-counts speed/power, which also drive motor mass.
5. **Energetics / cost of transport (todo).** Add electrical energy
   (copper loss ∝ τ²/Kt²) integrated over the gait as a third objective; larger
   motors have lower winding resistance, coupling sizing to efficiency.
6. **Mass feedback into dynamics (todo, high fidelity).** Chosen motor masses and
   rotor inertias should update link masses and joint `armature` in the sim, then
   re-evaluate — the "true" co-design loop. This invalidates a frozen policy, so it
   is most relevant to the in-training (gradient/Gumbel) approach.
7. **Co-optimize gear ratio (todo).** Trade peak torque vs rated speed vs reflected
   inertia per joint instead of fixing the transmission.

---

## Notes

- Gumbel-Softmax codesign remains in the repository for parallel investigation.
- This document only describes the current GA workflow used for batch design
  search and Pareto analysis.
- Keep this file updated whenever co-design selection logic, clustering
  thresholds, or output plots change.
