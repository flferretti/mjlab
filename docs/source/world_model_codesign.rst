.. _world_model_codesign:

World-Model Co-Design
=====================

``mjlab.world_model`` implements a **contact-enhanced, design-conditioned
world model** that amortizes actuator co-design evaluation: instead of paying
full simulation rollouts for every candidate design inside the NSGA-II loop
(``scripts/codesign_ga.py``), a single latent world model is trained once on
massively parallel simulation data in which *every environment carries a
different design*, and new designs are then scored in imagination — in
seconds per population.

How it works
------------

1. **Design space** (:class:`mjlab.world_model.DesignSpaceCfg`): a named,
   bounded vector of design parameters :math:`\theta` (v1: per-joint-group
   peak torque ``tau_max``, matching the GA genome exactly). It is the single
   write path for designs: the data collector's per-reset randomization, the
   evaluator's normalization, and GA-time injection all go through
   ``DesignSpaceCfg.apply``, which delegates to the motor-conditioned tasks'
   ``set_motor_tau_max``. An interpolation holdout set and an extrapolation
   shell are carved out at collection time and never trained on.

2. **Data collection** (``wm-collect``): rolls the frozen design-conditioned
   policy (the same one the GA evaluates with) across thousands of parallel
   environments, each with a per-reset resampled design, and records
   observations, actions, rewards (total + decomposed terms), applied joint
   torques, terminations, and privileged per-foot contact state
   (found/forces/air time) into time-major shards.

3. **World model** (``wm-train``): a decoder-free TD-MPC2-style latent
   dynamics model with SimNorm latents, FiLM-conditioned on a design
   embedding :math:`e(\theta)`. The *contact enhancement*: each ensemble
   member predicts next-step per-foot contact logits, and a straight-through
   Gumbel-sigmoid mode code gates the dynamics trunk (a soft mixture over
   contact modes, supervised by the contact sensors). Shared heads predict
   reward, termination, per-joint applied torque (for the GA's RMS thermal
   constraint), and the actor observation — so the *actual* frozen policy
   acts inside imagination, with the design and command observation slots
   overwritten exogenously. Training is pure device-agnostic PyTorch and runs
   on CUDA, ROCm (torch-rocm), or CPU; only collection needs the simulator.

4. **Evaluation** (``wm-eval``): scores held-out interpolation/extrapolation
   designs in imagination and (optionally) ground-truths them with true-sim
   rollouts, reporting Spearman rank correlation, pairwise ranking accuracy,
   and top-10 regret.

5. **Co-design** (``wm-codesign`` / ``--backend wm``): a surrogate backend
   for the NSGA-II loop. Every generation the world model scores the whole
   population; the top fraction by predicted performance plus any
   high-ensemble-uncertainty designs are re-scored in true simulation
   (``--wm-verify-topk``, default 0.25), and verified pairs feed an affine
   calibration of subsequent predictions. **Always re-evaluate the final
   front in true simulation** before reporting results.

Quickstart (QDD)
----------------

.. code-block:: bash

   # 1. Collect a design-randomized dataset (GPU, ~2-4 h at 4096 envs).
   uv run wm-collect --task Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond \
     --policy ./model_30000.pt --design-space qdd_v1 --out data/wm/qdd_v1 \
     --collect.num-envs 4096 --collect.steps 12500

   # 2. Train the world model (any device incl. ROCm; ~12 h default budget).
   uv run wm-train --data data/wm/qdd_v1 --train.logger wandb

   # 3. Gate: ranking quality on held-out designs (Spearman >= 0.9 interp).
   uv run wm-eval --checkpoint logs/world_model/qdd_v1/<run>/wm_final.pt \
     --policy ./model_30000.pt --split interp --true-sim

   # 4. Run co-design with the surrogate in the loop.
   uv run wm-codesign --wm-checkpoint logs/world_model/qdd_v1/<run>/wm_final.pt \
     --policy ./model_30000.pt --pop-size 32 --generations 25 \
     --multiobjective --cost-model powerlaw

The Go1 track (public robot, reproducible) uses
``Mjlab-Velocity-Flat-Unitree-Go1-MotorCond`` with ``--design-space go1_v1``;
train its motor-conditioned policy first with ``uv run train``.

Ablation knobs
--------------

The research claims are toggled purely through
:class:`mjlab.world_model.WorldModelCfg`:

- ``contact_heads=False, contact_gating=False`` — contact-blind baseline;
- ``contact_heads=True, contact_gating=False`` — auxiliary supervision only;
- ``contact_gating=True`` (default) — contact-mode-gated dynamics;
- ``privileged_encoder=True`` — sim-only upper bound (contact features as
  encoder inputs).

Notes
-----

- Warp (and therefore MuJoCo Warp) has no ROCm backend; collection and
  true-sim verification need an NVIDIA GPU or the (slow) Warp CPU fallback.
  ``wm-train``/``wm-eval`` (without ``--true-sim``) are pure PyTorch.
- Physical design parameters (rotor inertia via ``dof_armature``, motor-mass
  feedback into link inertials) are the planned v2 design space and follow
  the same ``DesignSpaceCfg`` write path through the domain-randomization
  model-field machinery.
