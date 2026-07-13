"""Configuration dataclasses for the contact-enhanced design-conditioned
world model.

The world model is a decoder-free latent dynamics model (TD-MPC2-style SimNorm
latents) whose dynamics and prediction heads are FiLM-conditioned on actuator
design parameters, with contact-mode-gated dynamics supervised by privileged
contact sensors. It is trained once on design-diverse simulation data and used
as an evaluation surrogate inside the actuator co-design loop
(``scripts/codesign_ga.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WorldModelCfg:
  """Architecture and loss configuration for the world model."""

  latent_dim: int = 512
  """Latent state dimension (must be divisible by ``simnorm_group_size``)."""
  simnorm_group_size: int = 8
  """SimNorm group size applied to encoder and dynamics outputs."""
  design_embed_dim: int = 64
  """Dimension of the design-parameter embedding e(theta)."""
  mode_embed_dim: int = 32
  """Dimension of the per-contact-point mode embedding."""
  hidden_dim: int = 512
  """Hidden width of all MLPs."""
  num_layers: int = 3
  """Number of hidden layers in trunk MLPs."""
  ensemble_size: int = 5
  """Number of dynamics+reward ensemble members (epistemic uncertainty)."""

  # -- Contact-enhancement mechanisms (the research knobs) -----------------
  contact_heads: bool = True
  """Auxiliary heads predicting current/next contact state and forces."""
  contact_gating: bool = True
  """Gate the dynamics trunk on the predicted next-step contact-mode code."""
  privileged_encoder: bool = False
  """Concatenate the privileged contact observation into the encoder input
  (sim-only upper-bound ablation; imagination then self-conditions on the
  contact head's prediction)."""
  conditioning: Literal["film"] = "film"
  """Design conditioning mechanism. The hypernetwork ablation arm is planned
  for the Phase-4 mechanism study."""

  gumbel_tau: float = 1.0
  """Temperature of the straight-through Gumbel-sigmoid on contact modes."""
  teacher_force_anneal_steps: int = 100_000
  """Gradient steps over which contact-mode teacher forcing anneals 1 -> 0."""

  # -- Training-loss weights ------------------------------------------------
  horizon: int = 32
  """Dynamics unroll length during training (sequence length is horizon+1)."""
  unroll_discount: float = 0.97
  """Per-step discount on unrolled losses."""
  w_consistency: float = 1.0
  w_reward: float = 1.0
  w_termination: float = 0.5
  w_contact_bce: float = 1.0
  w_contact_force: float = 0.5
  w_torque: float = 0.5
  w_obs: float = 0.5


@dataclass
class TrainCfg:
  """Optimization configuration for world-model training."""

  total_steps: int = 300_000
  batch_size: int = 1024
  """Number of sequences per batch (each of length ``horizon + 1``)."""
  learning_rate: float = 3e-4
  weight_decay: float = 1e-4
  grad_clip: float = 10.0
  ema_tau: float = 0.99
  """Momentum of the EMA target encoder (target <- tau*target + (1-tau)*online)."""
  seed: int = 0
  device: str | None = None
  """Torch device; ``None`` selects cuda if available, else cpu. All training
  code is device-agnostic PyTorch and runs on ROCm builds unchanged."""
  autocast: bool = True
  """bfloat16 autocast around model forward/backward (fp32 loss reduction)."""
  val_fraction: float = 0.05
  """Fraction of episodes held out for validation loss."""
  log_interval: int = 100
  val_interval: int = 2_000
  save_interval: int = 25_000
  logger: Literal["wandb", "tensorboard", "none"] = "none"
  wandb_project: str = "mjlab"
  run_name: str | None = None


@dataclass
class ImaginationCfg:
  """Configuration for imagination-based design evaluation."""

  rollout_steps: int = 200
  """Imagined steps per fitness rollout (matches the GA's rollout_steps)."""
  command: tuple[float, float, float] = (0.8, 0.0, 0.0)
  """Fixed twist command (vx, vy, wz) written into imagined observations,
  matching the GA's forward-walking evaluation."""
  ensemble_reduce: Literal["median", "mean"] = "median"
  """Reduction across ensemble members for the fitness estimate."""
  reset_on_termination: bool = True
  """Re-encode a fresh start state when the termination head fires (mirrors
  the env's auto-reset during true-sim GA rollouts)."""
  termination_threshold: float = 0.5


@dataclass
class CollectCfg:
  """Configuration for design-randomized data collection."""

  num_envs: int = 4096
  steps: int = 12_500
  """Env steps per environment (total transitions = num_envs * steps)."""
  seed: int = 0
  split: Literal["train", "interp", "extrap"] = "train"
  """Design-space region to sample: training interior (holdouts rejected),
  interpolation holdout designs, or the extrapolation shell."""
  n_holdout: int = 200
  """Number of interpolation holdout designs carved out of the interior."""
  policy_fraction: float = 0.85
  """Fraction of envs driven by the frozen policy (plus Gaussian noise)."""
  correlated_noise_fraction: float = 0.10
  """Fraction of envs driven by the policy plus temporally-correlated noise."""
  policy_noise_stds: tuple[float, ...] = (0.0, 0.1, 0.2)
  """Gaussian action-noise levels cycled across the policy-driven envs."""
  correlated_noise_std: float = 0.3
  correlated_noise_beta: float = 0.9
  """AR(1) coefficient of the temporally-correlated action noise."""
  flush_steps: int = 512
  """Steps buffered on-device between host transfers."""
  shard_steps: int = 2_048
  """Steps per on-disk shard file."""
  store_dtype: Literal["float16", "float32"] = "float16"


@dataclass
class WorldModelBackendCfg:
  """Configuration for the WM-in-the-loop co-design backend."""

  imagination: ImaginationCfg = field(default_factory=ImaginationCfg)
  verify_topk: float = 0.25
  """Fraction of each population re-scored in true sim (0 disables)."""
  verify_uncertainty_pct: float = 90.0
  """Designs whose ensemble-std exceeds this running percentile are also
  verified regardless of rank."""
  calibrate: bool = True
  """Fit an affine correction (pred -> true) from verified pairs."""
  min_calibration_pairs: int = 8
