"""Multi-objective actuator co-design for a humanoid via a frozen, motor-conditioned policy.

This optimizes, per *symmetric joint group*, (a) whether a custom high-torque
actuator is integrated and (b) the actuator's max torque limit, trading off task
performance against hardware cost / mass. It reuses a single pre-trained AMP-PPO
policy that ingests the current per-joint max torques as part of its observation,
so each genome is scored by a *rollout* (no retraining) on a massively parallel
GPU batch.

Design phases this builds on:
  * A reset event randomizes per-joint ``tau_max`` during training (motor-
    conditioned policy). For evaluation that event is disabled and the GA drives
    ``tau_max`` deterministically via ``set_motor_tau_max``.
  * The same ``set_motor_tau_max`` entry point writes both the observation buffer
    and the actuator effort limits, so the policy "sees" the design and the
    physics clips torque accordingly.

Example:
    uv run python scripts/codesign_ga.py \
        --backend mjlab \
        --task Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond \
        --policy /path/to/exported/policy.pt \
        --pop-size 32 --n-seeds 4 --rollout-steps 200 --generations 25
"""

from __future__ import annotations

import abc
import argparse
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import torch
from codesign_motor_model import MotorCostModel
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# 1. Humanoid domain: symmetric actuator groups + per-group catalog.
# ---------------------------------------------------------------------------
# Each group maps to one or more *physical* joints that must share an identical
# actuator (bilateral symmetry: left/right always tied). Torso/neck joints, if
# any, are their own single-joint groups (independent). Grouping shrinks the
# search space and enforces a laterally symmetric, physically viable design.


@dataclass(frozen=True)
class ActuatorGroup:
  """A set of joints that share one actuator size / torque limit."""

  name: str
  joints: tuple[str, ...]
  """Physical joints in this group (2 for an L/R pair, 1 for an independent joint)."""
  standard_tau: float
  """Catalog torque (Nm) used when no custom actuator is integrated."""
  tau_bounds: tuple[float, float]
  """(min, max) torque limit (Nm) searched when a custom actuator is integrated."""


# QDD lower-body humanoid: 6 symmetric leg groups with wide continuous search bounds
# (no discrete "custom" flag; all are continuous torque optimizations).
# Search space expanded to find diverse motor combinations that minimize peak torques.
TORQUE_CLUSTER_THRESHOLD_NM = 12.0

QDD_GROUPS: list[ActuatorGroup] = [
  ActuatorGroup("hip_pitch", ("l_hip_pitch", "r_hip_pitch"), 80.0, (30.0, 150.0)),
  ActuatorGroup("hip_roll", ("l_hip_roll", "r_hip_roll"), 80.0, (30.0, 150.0)),
  ActuatorGroup("hip_yaw", ("l_hip_yaw", "r_hip_yaw"), 40.0, (10.0, 100.0)),
  ActuatorGroup(
    "knee", ("l_knee", "r_knee"), 80.0, (30.0, 180.0)
  ),  # Increased to Type 3 capacity
  ActuatorGroup("ankle_pitch", ("l_ankle_pitch", "r_ankle_pitch"), 50.0, (15.0, 120.0)),
  ActuatorGroup("ankle_roll", ("l_ankle_roll", "r_ankle_roll"), 17.0, (5.0, 80.0)),
]

# Canonical joint order the policy/observation expects (must match the trained env).
QDD_JOINT_ORDER: list[str] = [
  "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll",
  "l_hip_yaw", "r_hip_yaw", "l_knee", "r_knee",
  "l_ankle_pitch", "r_ankle_pitch", "l_ankle_roll", "r_ankle_roll",
]  # fmt: skip

# Normalization range the policy was trained with (must match the env's obs term).
TAU_OBS_RANGE: tuple[float, float] = (5.0, 90.0)
WALK_SEED_GENOME: dict[str, float] = {
  "tau_hip_pitch": 31.98331642150879,
  "tau_hip_roll": 31.128311157226562,
  "tau_hip_yaw": 59.416717529296875,
  "tau_knee": 177.12530517578125,
  "tau_ankle_pitch": 47.57539749145508,
  "tau_ankle_roll": 29.87046241760254,
}

# ---------------------------------------------------------------------------
# Gene01 (nowrist_noneck) whole-body humanoid — 22 actuated joints.
# The motor-conditioned stairs policy ingests a 22-dim ``motor_tau_max`` block
# normalized over ``GENE_TAU_OBS_RANGE``. Joint order MUST match the training
# obs term (``GENE01_NOWRIST_NONECK_ACTUATED_JOINTS`` with ``preserve_order``),
# because the tau buffer, the observation and the effort limits are all keyed by
# this ordering. See gb_rl_locomotion assets/gene01_nowrist_noneck.py.
GENE_JOINT_ORDER: list[str] = [
  "l_hip_pitch", "r_hip_pitch", "torso_yaw",
  "l_hip_roll", "r_hip_roll", "torso_roll",
  "l_hip_yaw", "r_hip_yaw",
  "l_shoulder_pitch", "r_shoulder_pitch",
  "l_knee", "r_knee",
  "l_shoulder_roll", "r_shoulder_roll",
  "l_ankle_motor_1", "l_ankle_motor_2",
  "r_ankle_motor_1", "r_ankle_motor_2",
  "l_shoulder_yaw", "r_shoulder_yaw",
  "l_elbow", "r_elbow",
]  # fmt: skip

# Absolute torque range (Nm) the gene motor-conditioned policy was trained with
# (gb_rl_locomotion .../gene01_nowrist_noneck/motor_cond_*_env_cfg.py MOTOR_TAU_RANGE).
GENE_TAU_OBS_RANGE: tuple[float, float] = (40.0, 200.0)

# Symmetric actuator groups (L/R tied); torso joints are independent single-joint
# groups. Search bounds are clamped to the trained ``GENE_TAU_OBS_RANGE`` so the
# frozen policy only ever sees in-distribution tau conditioning.
GENE_GROUPS: list[ActuatorGroup] = [
  ActuatorGroup("hip_pitch", ("l_hip_pitch", "r_hip_pitch"), 120.0, GENE_TAU_OBS_RANGE),
  ActuatorGroup("hip_roll", ("l_hip_roll", "r_hip_roll"), 120.0, GENE_TAU_OBS_RANGE),
  ActuatorGroup("hip_yaw", ("l_hip_yaw", "r_hip_yaw"), 120.0, GENE_TAU_OBS_RANGE),
  ActuatorGroup("knee", ("l_knee", "r_knee"), 120.0, GENE_TAU_OBS_RANGE),
  ActuatorGroup(
    "ankle_motor_1", ("l_ankle_motor_1", "r_ankle_motor_1"), 120.0, GENE_TAU_OBS_RANGE
  ),
  ActuatorGroup(
    "ankle_motor_2", ("l_ankle_motor_2", "r_ankle_motor_2"), 120.0, GENE_TAU_OBS_RANGE
  ),
  ActuatorGroup(
    "shoulder_pitch",
    ("l_shoulder_pitch", "r_shoulder_pitch"),
    120.0,
    GENE_TAU_OBS_RANGE,
  ),
  ActuatorGroup(
    "shoulder_roll", ("l_shoulder_roll", "r_shoulder_roll"), 120.0, GENE_TAU_OBS_RANGE
  ),
  ActuatorGroup(
    "shoulder_yaw", ("l_shoulder_yaw", "r_shoulder_yaw"), 120.0, GENE_TAU_OBS_RANGE
  ),
  ActuatorGroup("elbow", ("l_elbow", "r_elbow"), 120.0, GENE_TAU_OBS_RANGE),
  ActuatorGroup("torso_yaw", ("torso_yaw",), 120.0, GENE_TAU_OBS_RANGE),
  ActuatorGroup("torso_roll", ("torso_roll",), 120.0, GENE_TAU_OBS_RANGE),
]

GENE_TASK_DEFAULT = "gb_rl_locomotion-amp-gene01_nowrist_noneck-motorcond-play-rough-V0"


@dataclass
class CodesignConfig:
  """Problem-level configuration."""

  groups: list[ActuatorGroup]
  joint_order: list[str]
  tau_obs_range: tuple[float, float] = TAU_OBS_RANGE
  # Hardware-cost weights (objective 2). Soft constraint to prefer 2-3 motor types.
  w_count: float = 0.5  # Diversity penalty: cost per unique motor type considered
  w_torque: float = 1.0  # Capacity penalty: cost per unit of total torque
  tau_ref: float = 90.0
  """Reference torque (Nm) used to normalize the cumulative-torque term."""
  w_motor_type_penalty: float = 50.0
  """Penalty multiplier for motor types above 3. Extremely strong to enforce 2-3 types.
  
  Penalty = max(0, n_types - 3) * w_motor_type_penalty
  
  Examples with w=50:
    n_types=2: penalty = 0        (cost ≈ 10-15)
    n_types=3: penalty = 0        (cost ≈ 10-15)
    n_types=4: penalty = 50       (cost ≈ 60-65)
    n_types=5: penalty = 100      (cost ≈ 110-115)
    n_types=6: penalty = 150      (cost ≈ 160-165)
  
  This makes designs with >3 types Pareto-dominated, as the ~0.05 reward difference
  cannot overcome 50-150 cost difference.
  """
  cost_model: MotorCostModel | None = None
  """Physical actuator mass/cost model (see ``codesign_motor_model``).

  When set, the capacity term uses the model's estimated total actuator mass (kg)
  instead of the legacy linear ``cum_tau / tau_ref`` proxy. ``None`` preserves the
  original behavior. A ``linear`` model reproduces the legacy proxy up to the
  anchored kg/Nm constant (numerically almost identical), while ``powerlaw`` /
  ``catalog`` capture the sub-linear mass-vs-torque scaling of real BLDC/QDD units.
  """
  rms_peak_ratio: float = 0.25
  """Thermal actuator-sizing limit: per-joint RMS torque must stay <=
  ``rms_peak_ratio`` * peak torque (a motor's continuous rating is ~1/4 its peak
  for BLDC/QDD). Measured from the rollout torque trace and enforced as a hard
  inequality constraint (multiobjective) or a strong penalty (single-objective).
  Set <= 0 to disable.
  """
  w_rms_penalty: float = 50.0
  """Single-objective penalty per Nm of RMS-over-limit (summed over joints).
  Strong, mirroring ``w_motor_type_penalty``, so RMS-violating designs are
  dominated regardless of their reward advantage."""

  def capacity_cost(self, tau_per_joint: np.ndarray, cum_tau: float) -> float:
    """Normalized hardware-capacity cost term (dimensionless).

    With a ``cost_model`` this is the total actuator mass (kg); otherwise it falls
    back to the legacy ``cum_tau / tau_ref`` proxy. Both are of comparable
    magnitude for the gene design (~15-40), so ``w_torque`` stays interpretable.
    """
    if self.cost_model is not None:
      return self.cost_model.design_mass(tau_per_joint)
    return cum_tau / self.tau_ref


# ---------------------------------------------------------------------------
# 2. Genome <-> design decoding.
# ---------------------------------------------------------------------------
# Pure continuous genome, per group g:
#   tau_g in [lo_g, hi_g]: max torque limit (Nm) for joint group g.
# (No discrete custom flag; all variables are continuous for richer search space)


def group_var_names(group: ActuatorGroup) -> tuple[str]:
  """Return only the torque variable name (no custom flag)."""
  return (f"tau_{group.name}",)


def _cluster_group_torques(genome: dict[str, float], cfg: CodesignConfig) -> np.ndarray:
  """Cluster per-group torques using the configured Nm threshold."""
  taus = np.array([float(genome[f"tau_{group.name}"]) for group in cfg.groups]).reshape(
    -1, 1
  )
  if len(taus) <= 1:
    return np.ones(len(taus), dtype=int)
  distances = pdist(taus, metric="euclidean")
  linkage_matrix = linkage(distances, method="complete")
  return fcluster(linkage_matrix, t=TORQUE_CLUSTER_THRESHOLD_NM, criterion="distance")


def count_motor_types(genome: dict[str, float], cfg: CodesignConfig) -> int:
  """Count distinct motor types using hierarchical clustering.

  Uses hierarchical agglomerative clustering with a distance threshold to adaptively
  group similar torques into the same motor type. This is more flexible than fixed
  rounding because it:
    1. Groups similar values regardless of their absolute values
    2. Counts "natural" clusters rather than arbitrary thresholds
    3. Scales better across the full [5, 90] Nm range

  Distance threshold: 12.0 Nm means torques within roughly one catalog step of
  each other cluster, so near-identical values do not split into fake types.

  Examples:
    τ = [36.0, 31.0, 25.0]
    → With threshold 12.0:
       * 36.0, 31.0, 25.0 cluster together
    → count: 1 type (no fake type splitting)

    τ = [90.0, 70.0, 50.0, 40.0, 30.0, 20.0]
    → Larger separations still form distinct clusters
    → count: multiple types

  Algorithm:
    1. Build hierarchical clustering dendrogram of 6 tau values
    2. Cut dendrogram at distance threshold (12.0 Nm)
    3. Count resulting clusters
  """
  return len(np.unique(_cluster_group_torques(genome, cfg)))


def decode_individual(
  genome: dict[str, float], cfg: CodesignConfig
) -> tuple[np.ndarray, int, float]:
  """Decode one genome dict into a per-joint torque vector + cost terms.

  Returns:
    tau_per_joint: (n_joints,) array in ``cfg.joint_order`` order (Nm).
    n_motor_choices: count of unique torque configurations (for diversity metric).
    cumulative_tau: sum of max torque over all physical joints (Nm).
  """
  joint_tau: dict[str, float] = {}
  cumulative = 0.0
  n_choices = 0  # Count distinct torque values
  seen_taus = set()

  for group in cfg.groups:
    (tau_name,) = group_var_names(group)
    tau = float(genome[tau_name])
    for j in group.joints:
      joint_tau[j] = tau
      cumulative += tau
    if tau not in seen_taus:
      n_choices += 1
      seen_taus.add(tau)

  tau_vec = np.array([joint_tau[j] for j in cfg.joint_order], dtype=np.float32)
  return tau_vec, n_choices, cumulative


def _group_design(genome: dict[str, float], cfg: CodesignConfig) -> dict[str, float]:
  return {g.name: round(float(genome[f"tau_{g.name}"]), 1) for g in cfg.groups}


def _per_joint_torques(tau_vec: np.ndarray, cfg: CodesignConfig) -> dict[str, float]:
  return {
    j: round(float(tau), 1)
    for j, tau in zip(cfg.joint_order, tau_vec.tolist(), strict=True)
  }


def _motor_selection_text(genome: dict[str, float], cfg: CodesignConfig) -> str:
  tau_vec, _, _ = decode_individual(genome, cfg)
  if cfg.cost_model is not None and cfg.cost_model.kind == "catalog":
    skus = cfg.cost_model.sku_names(tau_vec)
    counts = Counter(skus)
    return ", ".join(f"{n}x{sku}" for sku, n in sorted(counts.items()))

  labels = _cluster_group_torques(genome, cfg)
  cluster_groups: dict[int, list[str]] = {}
  cluster_taus: dict[int, list[float]] = {}
  for group, label in zip(cfg.groups, labels, strict=True):
    cluster_groups.setdefault(int(label), []).append(group.name)
    cluster_taus.setdefault(int(label), []).append(float(genome[f"tau_{group.name}"]))
  blocks: list[str] = []
  for idx, label in enumerate(sorted(cluster_groups), start=1):
    mean_tau = float(np.mean(cluster_taus[label]))
    names = ",".join(cluster_groups[label])
    blocks.append(f"type{idx}~{mean_tau:.1f}Nm[{names}]")
  return " | ".join(blocks)


def _print_final_solution_summary(
  title: str,
  genome: dict[str, float],
  cfg: CodesignConfig,
  reward: float,
  cost: float,
) -> None:
  tau_vec, _n_variants, cum_tau = decode_individual(genome, cfg)
  n_types = count_motor_types(genome, cfg)
  print(f"\n=== Final optimized design ({title}) ===")
  print(
    f"reward={reward:.3f} mass/cost={cost:.3f} cum_tau={cum_tau:.1f}Nm types={n_types}"
  )
  print(f"group_torques_nm={_group_design(genome, cfg)}")
  print(f"motor_selection={_motor_selection_text(genome, cfg)}")
  print(f"per_joint_torques_nm={_per_joint_torques(tau_vec, cfg)}")


# ---------------------------------------------------------------------------
# 3. Backend abstraction (mjlab / Isaac Lab) — isolates env + policy + rollout.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Performance metrics: mean reward, or AMP discriminator alignment.
# ---------------------------------------------------------------------------


class PerformanceMetric(abc.ABC):
  """Per-environment performance accumulated over a rollout."""

  @abc.abstractmethod
  def reset(self, num_envs: int, device: str) -> None: ...

  @abc.abstractmethod
  def update(
    self, backend: "CodesignBackend", obs: dict, reward: torch.Tensor
  ) -> None: ...

  @abc.abstractmethod
  def result(self) -> torch.Tensor:
    """(num_envs,) higher = better."""


class RewardMetric(PerformanceMetric):
  """Mean task reward over the rollout (motion-imitation proxy in AMP envs)."""

  def reset(self, num_envs: int, device: str) -> None:
    self._total = torch.zeros(num_envs, device=device)
    self._steps = 0

  def update(self, backend, obs, reward) -> None:
    self._total += reward
    self._steps += 1

  def result(self) -> torch.Tensor:
    return self._total / max(self._steps, 1)


class AmpAlignmentMetric(PerformanceMetric):
  """Mean AMP-discriminator "expert-likeness" over the rollout.

  Uses a frozen AMP discriminator (TorchScript) mapping an AMP observation (or a
  consecutive (prev, cur) transition) to a real logit; higher = more expert-like.
  We accumulate ``sigmoid(logit)`` per env per step, so the score is in [0, 1].
  This rewards designs that let the policy reproduce the reference motion,
  independent of task-reward shaping.
  """

  def __init__(self, discriminator: torch.nn.Module, use_transition: bool) -> None:
    self._disc = discriminator
    self._use_transition = use_transition

  def reset(self, num_envs: int, device: str) -> None:
    self._total = torch.zeros(num_envs, device=device)
    self._steps = 0
    self._prev_amp: torch.Tensor | None = None

  def update(self, backend, obs, reward) -> None:
    amp = backend._amp_obs(obs)
    if amp is None:
      raise RuntimeError(
        "AMP alignment requested but the env exposes no 'amp' observation group."
      )
    if self._use_transition:
      if self._prev_amp is None:
        self._prev_amp = amp
        return  # need two frames to form a transition
      feats = torch.cat([self._prev_amp, amp], dim=-1)
      self._prev_amp = amp
    else:
      feats = amp
    logit = self._disc(feats).squeeze(-1)
    self._total += torch.sigmoid(logit)
    self._steps += 1

  def result(self) -> torch.Tensor:
    return self._total / max(self._steps, 1)


class WalkabilityMetric(PerformanceMetric):
  """Command-tracking score plus height stability over the rollout."""

  def reset(self, num_envs: int, device: str) -> None:
    self._track_total = torch.zeros(num_envs, device=device)
    self._height_total = torch.zeros(num_envs, device=device)
    self._min_height = torch.full(
      (num_envs,), float("inf"), device=device, dtype=torch.float32
    )
    self._steps = 0

  def update(self, backend, obs, reward) -> None:
    actor_obs = backend._actor_obs(obs)
    target_vx = actor_obs[:, -15]
    self._track_total += -torch.abs(actor_obs[:, 0] - target_vx)
    height = backend._root_height()
    self._height_total += height
    self._min_height = torch.minimum(self._min_height, height)
    self._steps += 1

  def result(self) -> torch.Tensor:
    steps = max(self._steps, 1)
    mean_track = self._track_total / steps
    mean_height = self._height_total / steps
    return 2.0 * mean_track + mean_height + self._min_height


class RandomPolicy(torch.nn.Module):
  """Uniform random actions in [-1, 1] — for smoke-testing the GA plumbing."""

  def __init__(self, action_dim: int, device: str) -> None:
    super().__init__()
    self._dim = action_dim
    self._device = device

  def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
    n = actor_obs.shape[0]
    return torch.empty(n, self._dim, device=self._device).uniform_(-1.0, 1.0)


class FlatActorPolicy(torch.nn.Module):
  """Wrap an rsl_rl actor so it accepts a flat observation tensor."""

  def __init__(self, actor: torch.nn.Module) -> None:
    super().__init__()
    self.obs_normalizer = actor.obs_normalizer
    self.mlp = actor.mlp

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    latent = self.obs_normalizer(obs)
    return self.mlp(latent)


class OnnxPolicy(torch.nn.Module):
  """Wrap an ONNX policy for inference on mjlab.

  Handles dimension mismatch between simulator contracts:
  - motor-conditioned mjlab actor obs (180) vs IsaacLab ONNX blind+tau (177)
  - vanilla mjlab actor obs (168) vs blind ONNX (165)
  and falls back to padding/truncation for other mismatches.

  Example:
    policy = OnnxPolicy("policy_62000.onnx")
    obs = torch.zeros(32, 168)  # 32 envs, 168 obs dim (mjlab)
    actions = policy(obs)  # (32, 12)
  """

  def __init__(self, onnx_path: str, device: str = "cpu") -> None:
    super().__init__()
    import onnxruntime as rt

    self.onnx_path = onnx_path
    self.device = device
    self._sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    self._input_name = self._sess.get_inputs()[0].name
    self._output_name = self._sess.get_outputs()[0].name
    self._input_shape = self._sess.get_inputs()[0].shape

    # Some exported policies (e.g. BeyondMimic tracking) declare extra inputs
    # beyond the observation — most commonly a scalar ``time_step`` that only
    # drives auxiliary reference-motion outputs, NOT the action. onnxruntime
    # still requires every declared input, so feed zeros for the non-obs ones.
    self._extra_inputs = [
      (inp.name, [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape])
      for inp in self._sess.get_inputs()[1:]
    ]

    # ONNX expects [1, onnx_obs_dim], mjlab provides [batch, 168]
    # Extract expected obs dimension from ONNX model
    self._onnx_obs_dim = int(self._input_shape[1])
    print(
      f"[OnnxPolicy] Loaded {onnx_path}: expects obs_dim={self._onnx_obs_dim}, "
      f"batch_size={self._input_shape[0]}"
      + (
        f", extra_inputs={[n for n, _ in self._extra_inputs]}"
        if self._extra_inputs
        else ""
      )
    )

  def _adapt_obs(self, obs: torch.Tensor) -> np.ndarray:
    """Convert mjlab flat obs to ONNX input contract."""
    mjlab_obs_dim = int(obs.shape[-1])
    if mjlab_obs_dim == self._onnx_obs_dim:
      return obs.cpu().numpy().astype(np.float32)

    # Mjlab actor observations include base linear velocity as the leading 3 dims.
    # gb-rl / IsaacLab blind ONNX contracts do not include this block.
    if mjlab_obs_dim == self._onnx_obs_dim + 3:
      return obs[:, 3:].cpu().numpy().astype(np.float32)

    if mjlab_obs_dim < self._onnx_obs_dim:
      # Pad with zeros
      batch_size = obs.shape[0]
      padded = torch.zeros(
        batch_size, self._onnx_obs_dim, device=obs.device, dtype=obs.dtype
      )
      padded[:, :mjlab_obs_dim] = obs
      return padded.cpu().numpy().astype(np.float32)
    else:
      # Truncate (less likely but possible)
      return obs[:, : self._onnx_obs_dim].cpu().numpy().astype(np.float32)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """Inference on a batch of observations.

    Args:
      obs: (batch_size, 168) flat observation tensor (mjlab format)

    Returns:
      actions: (batch_size, 12) action tensor
    """
    batch_size = obs.shape[0]
    obs_adapted = self._adapt_obs(obs)

    # ONNX inference processes one at a time if shape is [1, ...]
    # For batches, reshape to (batch, obs_dim) or call in loop
    actions_list = []
    for i in range(batch_size):
      obs_single = obs_adapted[i : i + 1]  # (1, obs_dim)
      feed = {self._input_name: obs_single}
      for name, shape in self._extra_inputs:
        feed[name] = np.zeros(shape, dtype=np.float32)
      action = self._sess.run([self._output_name], feed)[0]
      actions_list.append(action[0])  # Remove batch dim

    actions_np = np.stack(actions_list, axis=0)
    return torch.from_numpy(actions_np).to(obs.device).to(obs.dtype)


def _load_mjlab_policy_module(
  task: str, policy_path: str, device: str
) -> torch.nn.Module:
  """Load a policy: ONNX, TorchScript, or rsl_rl checkpoint.

  Args:
    task: environment task name (used to load runner config)
    policy_path: path to .onnx, .pt, or .jit file
    device: "cpu" or "cuda"

  Returns:
    A torch.nn.Module that accepts flat observation tensors
  """
  # Try ONNX first (easiest, no environment needed)
  if policy_path.endswith(".onnx"):
    return OnnxPolicy(policy_path, device=device).eval()

  # Try TorchScript next
  try:
    return torch.jit.load(policy_path, map_location=device).eval()
  except (RuntimeError, ValueError) as exc:
    msg = str(exc)
    if "constants.pkl" not in msg and "PytorchStreamReader" not in msg:
      raise

  # Fall back to rsl_rl checkpoint (requires environment setup)
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.amp_runner import MjlabAmpOnPolicyRunner
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = 1
  rl_cfg = load_rl_cfg(task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapper = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  try:
    runner = MjlabAmpOnPolicyRunner(wrapper, asdict(rl_cfg), device=device)
    runner.load(policy_path, load_cfg={"actor": True}, strict=True)
    actor = runner.actor_critic.actor.to(device).eval()
    return FlatActorPolicy(actor).eval()
  finally:
    wrapper.close()


# ---------------------------------------------------------------------------
# 3. Backend abstraction (mjlab / Isaac Lab) — isolates env + policy + rollout.
# ---------------------------------------------------------------------------


class CodesignBackend(abc.ABC):
  """Owns the vectorized env and frozen policy; scores torque designs by rollout.

  Implementations must build the env WITHOUT the per-reset torque randomization
  event (so the GA controls ``tau_max``), expose the joint order, and provide a
  deterministic-policy rollout that accumulates a performance metric per env.
  """

  device: str
  num_envs: int
  joint_order: list[str]

  def __init__(self, rollout_steps: int, metric: PerformanceMetric) -> None:
    self.rollout_steps = rollout_steps
    self.metric = metric
    self._tau_full: torch.Tensor | None = None
    # Per-design RMS torque (L, n_joints), set by the last evaluate() call; None
    # if this backend cannot report applied torque (constraint then skipped).
    self.last_rms_LJ: np.ndarray | None = None
    self._rollout_rms: torch.Tensor | None = None

  @abc.abstractmethod
  def _reset(self) -> dict: ...

  @abc.abstractmethod
  def _step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor]: ...

  @abc.abstractmethod
  def _actor_obs(self, obs: dict) -> torch.Tensor: ...

  @abc.abstractmethod
  def _policy(self, actor_obs: torch.Tensor) -> torch.Tensor: ...

  @abc.abstractmethod
  def _root_height(self) -> torch.Tensor: ...

  @abc.abstractmethod
  def _write_tau(self, tau_full: torch.Tensor) -> None:
    """Write the (num_envs, n_joints) torque matrix to obs buffer + effort limits."""

  @abc.abstractmethod
  def action_dim(self) -> int:
    """Total action dimension (for the random smoke-test policy)."""

  def _applied_torque_LJ(self) -> torch.Tensor | None:
    """Applied joint torque (num_envs, n_actuated_joints) in ``joint_order``.

    Returns None if the backend cannot report per-joint torque (the RMS
    thermal constraint is then skipped for that backend).
    """
    return None

  def _amp_obs(self, obs: dict) -> torch.Tensor | None:
    """AMP observation group (used by AmpAlignmentMetric); None if unavailable."""
    return obs.get("amp") if isinstance(obs, dict) else None

  def evaluate(self, tau_LJ: torch.Tensor, n_seeds: int) -> np.ndarray:
    """Score a batch of L designs, each replicated over ``n_seeds`` envs.

    The population is mapped onto the env batch as ``L * n_seeds`` rollouts; the
    reward is averaged over seeds (and steps) to reduce reset-noise variance.
    """
    L = tau_LJ.shape[0]
    need = L * n_seeds
    assert need <= self.num_envs, (
      f"pop*seeds ({need}) exceeds env batch ({self.num_envs})"
    )

    # Replicate each design across its evaluation seeds, then pad the unused
    # tail of the (fixed-size) env batch with the last design.
    tau_rep = tau_LJ.repeat_interleave(n_seeds, dim=0)
    if need < self.num_envs:
      pad = tau_rep[-1:].expand(self.num_envs - need, -1)
      tau_full = torch.cat([tau_rep, pad], dim=0)
    else:
      tau_full = tau_rep
    self._tau_full = tau_full.contiguous()

    mean_reward = self._rollout()  # (num_envs,)
    per_design = mean_reward[:need].view(L, n_seeds).mean(dim=1)

    # Per-joint RMS torque over the rollout, averaged across seeds (L, n_joints).
    if self._rollout_rms is not None:
      rms = self._rollout_rms[:need].view(L, n_seeds, -1).mean(dim=1)
      self.last_rms_LJ = rms.detach().cpu().numpy()
    else:
      self.last_rms_LJ = None

    return per_design.detach().cpu().numpy()

  @torch.no_grad()
  def _rollout(self) -> torch.Tensor:
    assert self._tau_full is not None
    # Set torques BEFORE reset so the first observation reflects the design,
    # then reset to start fresh episodes for every env.
    self._write_tau(self._tau_full)
    obs = self._reset()
    self._write_tau(self._tau_full)  # reset events may touch limits; re-assert.

    self.metric.reset(self.num_envs, self.device)
    tau_sq_sum: torch.Tensor | None = None
    n_tau_steps = 0
    for _ in range(self.rollout_steps):
      actions = self._policy(self._actor_obs(obs))
      obs, reward = self._step(actions)
      self.metric.update(self, obs, reward)
      tau = self._applied_torque_LJ()  # (num_envs, n_joints) or None
      if tau is not None:
        tau_sq = tau.detach() ** 2
        tau_sq_sum = tau_sq if tau_sq_sum is None else tau_sq_sum + tau_sq
        n_tau_steps += 1
      # Re-assert each step so any auto-reset on termination keeps the design.
      self._write_tau(self._tau_full)
    self._rollout_rms = (
      torch.sqrt(tau_sq_sum / max(n_tau_steps, 1)) if tau_sq_sum is not None else None
    )
    return self.metric.result()


class MjlabBackend(CodesignBackend):
  """mjlab + MuJoCo-Warp backend."""

  def __init__(
    self,
    task: str,
    policy_path: str | None,
    num_envs: int,
    joint_order: list[str],
    tau_obs_range: tuple[float, float],
    rollout_steps: int,
    metric: PerformanceMetric,
    device: str = "cuda:0",
  ) -> None:
    super().__init__(rollout_steps, metric)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    self.device = device
    self.num_envs = num_envs
    self.joint_order = joint_order
    self._tau_obs_range = tau_obs_range

    try:
      cfg = load_env_cfg(task, play=True)
    except TypeError:
      cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    # Match deployment intent for locomotion co-design: forward-walking evaluation.
    if hasattr(cfg, "commands") and "twist" in cfg.commands:
      twist_cmd = cfg.commands["twist"]
      twist_cmd.ranges.lin_vel_x = (0.8, 0.8)
      twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
      twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
      if hasattr(twist_cmd, "rel_standing_envs"):
        twist_cmd.rel_standing_envs = 0.0
      if hasattr(twist_cmd, "rel_heading_envs"):
        twist_cmd.rel_heading_envs = 0.0
      if hasattr(twist_cmd, "heading_command"):
        twist_cmd.heading_command = False
      if hasattr(twist_cmd.ranges, "heading"):
        twist_cmd.ranges.heading = None
    # Disable play-time randomization so GA sees the same fixed deployment setup
    # as the sim2sim rollout.
    if getattr(cfg, "events", None) is not None:
      for name in (
        "encoder_bias",
        "base_com",
        "foot_friction_slide",
        "randomize_robot_mass",
        "randomize_actuator_gains",
        "randomize_joint_friction",
        "joint_default_pos_noise",
        "randomize_terrain",
      ):
        cfg.events.pop(name, None)
      cfg.events.pop("randomize_motor_tau_max", None)
      cfg.events.pop("push_robot", None)
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)

    # Frozen policy: a TorchScript module mapping actor-obs -> action (as exported
    # by the rsl_rl/amp pipeline). Falls back to a random policy for smoke tests.
    if policy_path is not None:
      self.policy_module = _load_mjlab_policy_module(task, policy_path, device)
    else:
      self.policy_module = RandomPolicy(self.action_dim(), device)

    from mjlab.tasks.velocity.mdp.motor_randomization import set_motor_tau_max

    self._set_tau = set_motor_tau_max
    self._env_ids = torch.arange(num_envs, device=device)
    # Joint ids into qfrc_actuator matching ``joint_order`` (for RMS torque).
    robot = self.env.unwrapped.scene["robot"]
    self._tau_joint_ids, _ = robot.find_joints(list(joint_order), preserve_order=True)

  def action_dim(self) -> int:
    return self.env.unwrapped.action_manager.total_action_dim

  def _reset(self) -> dict:
    obs, _ = self.env.reset()
    return obs

  def _step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor]:
    obs, reward, _terminated, _truncated, _info = self.env.step(actions)
    return obs, reward

  def _actor_obs(self, obs: dict) -> torch.Tensor:
    return obs["actor"]

  def _policy(self, actor_obs: torch.Tensor) -> torch.Tensor:
    return self.policy_module(actor_obs)

  def _root_height(self) -> torch.Tensor:
    robot = self.env.unwrapped.scene["robot"]
    return robot.data.root_link_pos_w[:, 2]

  def _applied_torque_LJ(self) -> torch.Tensor | None:
    robot = self.env.unwrapped.scene["robot"]
    return robot.data.qfrc_actuator[:, self._tau_joint_ids]

  def _write_tau(self, tau_full: torch.Tensor) -> None:
    self._set_tau(
      self.env.unwrapped,
      self._env_ids,
      tau_full,
      self.joint_order,
      self._tau_obs_range,
    )


def _clamp_stairs_step_height(env_cfg, max_step_height: float) -> None:
  """Clamp every stairs subterrain's step_height_range to <= max_step_height (m).

  No-op for flat envs (terrain_generator is None) or subterrains without a
  ``step_height_range`` field. Also disables terrain curriculum so difficulty
  can't scale the steps back above the cap.
  """
  gen = getattr(
    getattr(getattr(env_cfg, "scene", None), "terrain", None), "terrain_generator", None
  )
  if gen is None:
    return
  for sub in gen.sub_terrains.values():
    rng = getattr(sub, "step_height_range", None)
    if rng is not None:
      sub.step_height_range = (
        min(rng[0], max_step_height),
        min(rng[1], max_step_height),
      )
  gen.curriculum = False


class IsaacLabBackend(CodesignBackend):
  """Isaac Lab backend.

  Mirrors the mjlab backend; the only substantive differences are how the env is
  created (``gym.make``) and that actuator effort limits live in a dict
  (``articulation.actuators``) with ``effort_limit``/``joint_names`` — handled by
  the gb_rl_locomotion ``set_motor_tau_max`` helper that ships with the
  motor-conditioned task.
  """

  def __init__(
    self,
    task: str,
    policy_path: str | None,
    num_envs: int,
    joint_order: list[str],
    tau_obs_range: tuple[float, float],
    rollout_steps: int,
    metric: PerformanceMetric,
    device: str = "cuda:0",
    max_step_height: float | None = None,
  ) -> None:
    super().__init__(rollout_steps, metric)
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg  # type: ignore

    self.device = device
    self.num_envs = num_envs
    self.joint_order = joint_order
    self._tau_obs_range = tau_obs_range

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    # Disable training-time torque randomization (configclass attr, not a dict).
    if hasattr(env_cfg.events, "randomize_motor_tau_max"):
      env_cfg.events.randomize_motor_tau_max = None
    # Cap the stairs step height so the RMS/torque demand reflects the real
    # deployment envelope (e.g. 0.10 m) rather than the training max (0.23 m).
    if max_step_height is not None:
      _clamp_stairs_step_height(env_cfg, max_step_height)
    # Evaluate every design under the same fixed forward-walking command so the
    # reward score reflects the design, not the sampled command.
    cmd = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if cmd is not None:
      cmd.ranges.lin_vel_x = (0.8, 0.8)
      cmd.ranges.lin_vel_y = (0.0, 0.0)
      cmd.ranges.ang_vel_z = (0.0, 0.0)
      if hasattr(cmd, "rel_standing_envs"):
        cmd.rel_standing_envs = 0.0
      if hasattr(cmd, "rel_heading_envs"):
        cmd.rel_heading_envs = 0.0
      if hasattr(cmd, "heading_command"):
        cmd.heading_command = False
    self.env = gym.make(task, cfg=env_cfg)

    if policy_path is None:
      self.policy_module = RandomPolicy(self.action_dim(), device)
    elif policy_path.endswith(".onnx"):
      # Reuse the exported ONNX directly (no jit re-export). Inference runs on
      # CPU per env; fine for modest populations. The gene stairs ONNX obs dim
      # (147) matches the env "policy" group exactly, so OnnxPolicy passes it
      # through unchanged.
      self.policy_module = OnnxPolicy(policy_path, device=device).eval()
    else:
      self.policy_module = torch.jit.load(policy_path, map_location=device).eval()

    from gb_rl_locomotion.mdp.motor_randomization import (  # type: ignore
      set_motor_tau_max,
    )
    from isaaclab.managers import SceneEntityCfg  # type: ignore

    self._set_tau = set_motor_tau_max
    # The tau buffer / obs term / effort limits are all keyed by this ordered
    # actuated-joint selector; it MUST match ``joint_order`` (see GENE_JOINT_ORDER).
    self._motor_asset_cfg = SceneEntityCfg(
      "robot", joint_names=list(joint_order), preserve_order=True
    )
    self._env_ids = torch.arange(num_envs, device=device)
    # Joint ids into applied_torque matching ``joint_order`` (for RMS torque).
    robot = self.env.unwrapped.scene["robot"]
    self._tau_joint_ids, _ = robot.find_joints(list(joint_order), preserve_order=True)

  def action_dim(self) -> int:
    return int(self.env.unwrapped.action_manager.total_action_dim)

  def _reset(self) -> dict:
    obs, _ = self.env.reset()
    return obs

  def _step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor]:
    obs, reward, _terminated, _truncated, _info = self.env.step(actions)
    return obs, reward

  def _actor_obs(self, obs: dict) -> torch.Tensor:
    return obs["policy"]

  def _policy(self, actor_obs: torch.Tensor) -> torch.Tensor:
    return self.policy_module(actor_obs)

  def _root_height(self) -> torch.Tensor:
    robot = self.env.unwrapped.scene["robot"]
    return robot.data.root_link_pos_w[:, 2]

  def _applied_torque_LJ(self) -> torch.Tensor | None:
    robot = self.env.unwrapped.scene["robot"]
    return robot.data.applied_torque[:, self._tau_joint_ids]

  def _write_tau(self, tau_full: torch.Tensor) -> None:
    self._set_tau(
      self.env.unwrapped,
      self._env_ids,
      tau_full,
      self._motor_asset_cfg,
      self._tau_obs_range,
    )


class Sim2SimBackend:
  """Direct sim2sim backend using the validated MuJoCo-C rollout."""

  def __init__(
    self,
    task: str,
    policy_path: str | None,
    num_envs: int,
    joint_order: list[str],
    tau_obs_range: tuple[float, float],
    rollout_steps: int,
    metric: PerformanceMetric,
    device: str = "cpu",
  ) -> None:
    del task, num_envs, device
    if policy_path is None:
      raise ValueError("sim2sim backend requires an ONNX policy path")

    from sim2sim import Sim2SimRunner

    self.rollout_steps = rollout_steps
    self.metric = metric
    self.device = "cpu"
    self._tau_obs_range = tau_obs_range
    self.joint_order = list(joint_order)
    self.runner = Sim2SimRunner(
      onnx_path=policy_path,
      obs_layout="auto",
      tau_obs_range=tau_obs_range,
    )
    self.num_envs = 1
    self._tau_full: torch.Tensor | None = None

  def _apply_tau(self, tau_vec: np.ndarray) -> None:
    tau = tau_vec.astype(np.float32)
    self.runner.effort_limit[:] = tau
    self.runner.motor_tau_obs[:] = np.clip(
      (tau - self._tau_obs_range[0])
      / (self._tau_obs_range[1] - self._tau_obs_range[0]),
      0.0,
      1.0,
    )
    for i, aid in enumerate(self.runner.actuator_idx):
      self.runner.model.actuator_forcerange[aid] = (-tau[i], tau[i])

  def evaluate(self, tau_LJ: torch.Tensor, n_seeds: int) -> np.ndarray:
    scores: list[float] = []
    tau_rows = tau_LJ.detach().cpu().numpy()
    for tau_vec in tau_rows:
      seed_scores = []
      for _ in range(n_seeds):
        seed_scores.append(self._rollout_one(tau_vec))
      scores.append(float(np.mean(seed_scores)))
    return np.asarray(scores, dtype=np.float64)

  def _rollout_one(self, tau_vec: np.ndarray) -> float:
    self._apply_tau(tau_vec)
    self.runner.reset()
    self.runner.command[:] = [0.8, 0.0, 0.0]
    self.runner.filtered_command[:] = 0.0

    lin_vx = []
    heights = []
    for _ in range(self.rollout_steps):
      self.runner._step_once()
      base_quat = self.runner.data.qpos[3:7]
      rot = Rotation.from_quat(base_quat, scalar_first=True)
      v_body = rot.apply(self.runner.data.qvel[0:3], inverse=True).astype(np.float32)
      lin_vx.append(float(v_body[0]))
      heights.append(float(self.runner.data.qpos[2]))
    lin_vx_np = np.asarray(lin_vx, dtype=np.float32)
    heights_np = np.asarray(heights, dtype=np.float32)
    mean_vx = float(lin_vx_np.mean())
    mean_height = float(heights_np.mean())
    min_height = float(heights_np.min())
    del min_height
    return float(mean_vx + 0.1 * mean_height + 0.1 * float(heights_np.min()))


def build_backend(name: str, **kwargs) -> CodesignBackend:
  """Factory: switch simulation backend with a single argument."""
  if name == "mjlab":
    return MjlabBackend(**kwargs)
  if name == "isaaclab":
    return IsaacLabBackend(**kwargs)
  if name == "sim2sim":
    return Sim2SimBackend(**kwargs)
  if name == "wm":
    return _build_wm_backend(**kwargs)
  raise ValueError(
    f"Unknown backend '{name}' (expected 'mjlab', 'isaaclab', 'sim2sim', or 'wm')."
  )


def _build_wm_backend(
  wm_checkpoint: str | None = None,
  wm_verify_topk: float = 0.25,
  **kwargs,
) -> CodesignBackend:
  """World-model surrogate backend (mjlab.world_model), duck-typed to
  CodesignBackend. With ``wm_verify_topk > 0`` an MjlabBackend is built as the
  true-sim verifier for the top/uncertain slice of each generation. The final
  reported front must be re-evaluated in true sim (rerun with --backend mjlab
  on the winning designs)."""
  from mjlab.world_model import WorldModelBackend, WorldModelBackendCfg

  if wm_checkpoint is None:
    raise ValueError("--wm-checkpoint is required with --backend wm.")
  verifier = MjlabBackend(**kwargs) if wm_verify_topk > 0.0 else None
  backend = WorldModelBackend.from_checkpoint(
    checkpoint=wm_checkpoint,
    task=kwargs["task"],
    policy_path=kwargs["policy_path"],
    joint_order=kwargs["joint_order"],
    rollout_steps=kwargs["rollout_steps"],
    device=kwargs["device"],
    metric=kwargs["metric"],
    verifier=verifier,
    cfg=WorldModelBackendCfg(verify_topk=wm_verify_topk),
    num_envs=kwargs["num_envs"],
  )
  return cast(CodesignBackend, backend)


# ---------------------------------------------------------------------------
# 4. pymoo mixed-variable, multi-objective problem (vectorized GPU evaluation).
# ---------------------------------------------------------------------------


def _make_problem(
  backend: CodesignBackend,
  cfg: CodesignConfig,
  n_seeds: int,
  multiobjective: bool = False,
  max_motor_types: int = 3,
):
  # Pure continuous variables: per group, optimize torque limit (Nm)
  # Ordered dict preserves variable order for encoding/decoding
  from collections import OrderedDict

  from pymoo.core.problem import Problem
  from pymoo.core.variable import Real

  variables = OrderedDict()
  for group in cfg.groups:
    (tau_name,) = group_var_names(group)
    variables[tau_name] = Real(bounds=group.tau_bounds)

  class CodesignProblem(Problem):
    def __init__(self) -> None:
      # RMS thermal constraint adds one inequality (per-design worst-joint
      # RMS-over-limit) when enabled.
      rms_active = cfg.rms_peak_ratio > 0.0
      if multiobjective:
        # True bi-objective front: minimize (-performance, actuator mass). NSGA-II
        # returns the whole non-dominated set in ONE run (no w_torque scalarization
        # or sweep). The motor-diversity limit is a hard inequality constraint so
        # designs are pushed toward <= max_motor_types distinct actuators; the RMS
        # limit (RMS <= rms_peak_ratio * peak torque) is a second hard constraint.
        super().__init__(vars=variables, n_obj=2, n_ieq_constr=1 + int(rms_active))
      else:
        # Single scalarized objective: performance - cost
        # Increasing w_count/w_torque directly prefers cheaper designs
        super().__init__(vars=variables, n_obj=1)

    def _evaluate(self, X, out, *args, **kwargs):
      # X is a 1-D object array of genome dicts (continuous variables only).
      # Decode the WHOLE population, run a single batched rollout, then split.
      tau_rows, costs, n_motor_types = [], [], []
      for genome in X:
        tau_vec, n_choices, cum_tau = decode_individual(genome, cfg)
        tau_rows.append(tau_vec)
        costs.append((n_choices, cum_tau))
        n_motor_types.append(count_motor_types(genome, cfg))

      tau_LJ = torch.as_tensor(
        np.stack(tau_rows), dtype=torch.float32, device=backend.device
      )
      performance = backend.evaluate(tau_LJ, n_seeds)  # (L,)
      rms_active = cfg.rms_peak_ratio > 0.0
      rms_LJ = backend.last_rms_LJ  # (L, n_joints) Nm, or None if unavailable

      def _rms_overload(i: int, tau_vec: np.ndarray) -> float:
        """Worst-joint RMS-over-limit in Nm (>0 = thermal violation)."""
        if not rms_active or rms_LJ is None:
          return 0.0
        return float(np.max(rms_LJ[i] - cfg.rms_peak_ratio * tau_vec))

      if multiobjective:
        # Objective 1: -reward (minimized). Objective 2: total actuator mass/cost.
        # Constraint 1: n_types - max_motor_types <= 0.
        # Constraint 2 (if enabled): worst-joint (RMS - ratio*peak) <= 0.
        f_rows, g_rows = [], []
        for i, (perf, tau_vec, (_n_choices, cum_tau), n_types) in enumerate(
          zip(performance, tau_rows, costs, n_motor_types, strict=True)
        ):
          mass_cost = cfg.capacity_cost(tau_vec, cum_tau)
          f_rows.append([-float(perf), float(mass_cost)])
          g = [float(n_types) - float(max_motor_types)]
          if rms_active:
            g.append(_rms_overload(i, tau_vec))
          g_rows.append(g)
        out["F"] = np.array(f_rows, dtype=np.float64)
        out["G"] = np.array(g_rows, dtype=np.float64)
        return

      # Scalarized objective: maximize performance minus hardware cost
      f_obj = []
      for i, (perf, tau_vec, (n_choices, cum_tau), n_types) in enumerate(
        zip(performance, tau_rows, costs, n_motor_types, strict=True)
      ):
        # Base cost: diversity (number of unique motor models) + total capacity.
        # ``capacity_cost`` uses the physical mass model when configured, else the
        # legacy ``cum_tau / tau_ref`` proxy.
        base_cost = cfg.w_count * n_choices + cfg.w_torque * cfg.capacity_cost(
          tau_vec, cum_tau
        )

        # STRONG penalty: strongly discourage >3 types so designs naturally
        # evolve toward 2-3 types. Must exceed typical reward variance.
        # E.g., if reward ranges ~0.05-0.1 and cost ~5-15, penalty must be ~10+ to
        # make "4 types" designs worse than "3 types" designs.
        excess_penalty = max(0.0, float(n_types - 3)) * cfg.w_motor_type_penalty

        # STRONG thermal penalty: each Nm of RMS-over-limit costs w_rms_penalty,
        # so a design whose motors would thermally overload is dominated.
        rms_penalty = cfg.w_rms_penalty * max(0.0, _rms_overload(i, tau_vec))
        total_cost = base_cost + excess_penalty + rms_penalty

        # Fitness: higher performance is better, lower cost is better
        # Minimize: -performance + cost
        fitness = -perf + total_cost
        f_obj.append(fitness)

      out["F"] = np.array(f_obj, dtype=np.float64).reshape(-1, 1)

  return CodesignProblem()


def run_optimization(
  backend: CodesignBackend,
  cfg: CodesignConfig,
  pop_size: int,
  n_seeds: int,
  generations: int,
  seed: int = 0,
  seed_genome: dict[str, float] | None = None,
  multiobjective: bool = False,
  max_motor_types: int = 3,
):
  from pymoo.algorithms.moo.nsga2 import NSGA2
  from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
  )
  from pymoo.core.sampling import Sampling
  from pymoo.optimize import minimize

  problem = _make_problem(
    backend,
    cfg,
    n_seeds,
    multiobjective=multiobjective,
    max_motor_types=max_motor_types,
  )

  class SeededMixedVariableSampling(Sampling):
    def __init__(self, seed_genome: dict[str, float] | None) -> None:
      super().__init__()
      self._seed_genome = seed_genome or {}

    def _do(self, problem, n_samples, random_state=None, **kwargs):
      X = MixedVariableSampling()._do(
        problem, n_samples, random_state=random_state, **kwargs
      )
      if not self._seed_genome or n_samples == 0:
        return X

      seed = {}
      for name in problem.vars.keys():
        value = self._seed_genome.get(name)
        if value is None:
          value = X[0][name]
        seed[name] = float(value)
      X[0] = seed
      return X

  # NSGA-II with MixedVariableSampling to handle dict-based continuous variables
  # MixedVariableSampling works for pure continuous dict variables too
  sampling = (
    SeededMixedVariableSampling(seed_genome)
    if seed_genome is not None
    else MixedVariableSampling()
  )
  algorithm = NSGA2(
    pop_size=pop_size,
    n_offsprings=pop_size,
    sampling=sampling,
    mating=MixedVariableMating(
      eliminate_duplicates=MixedVariableDuplicateElimination()
    ),
    eliminate_duplicates=MixedVariableDuplicateElimination(),
  )

  result = minimize(
    problem,
    algorithm,
    termination=("n_gen", generations),
    seed=seed,
    verbose=True,
    save_history=False,
  )
  return result


# ---------------------------------------------------------------------------
# 5. Reporting.
# ---------------------------------------------------------------------------


def report_pareto(result, cfg: CodesignConfig, out_path: str | None) -> None:
  if result.X is None or result.F is None:
    print(
      "\n⚠️  No feasible design found (all violated the motor-type constraint). "
      "Try raising --max-motor-types, increasing --generations/--pop-size, or "
      "widening TORQUE_CLUSTER_THRESHOLD_NM."
    )
    return
  X = np.atleast_1d(result.X)
  F = np.atleast_2d(result.F)

  # Validate result shape
  if F.ndim != 2 or F.shape[1] < 1:
    print(f"\n⚠️  Unexpected result shape: F.shape={F.shape}")
    print(f"Expected (n_designs, 1+) objectives, got {F.shape}")
    return

  if F.shape[1] >= 2:
    # Bi-objective front: F[:,0] = -reward, F[:,1] = actuator mass/cost.
    # Report the non-dominated set from best reward to cheapest.
    order = np.argsort(F[:, 0])
    print("\n=== Pareto front (bi-objective: reward vs mass) ===")
    print(f"{'reward':>8} {'mass/cost':>10} {'types':>6} {'cumTau(Nm)':>11}  design")
    for i in order:
      genome = X[i] if isinstance(X[i], dict) else dict(X[i])
      tau_vec, _n_variants, cum_tau = decode_individual(genome, cfg)
      n_motor_types = count_motor_types(genome, cfg)
      design = {
        g.name: round(float(tau_vec[cfg.joint_order.index(g.joints[0])]), 1)
        for g in cfg.groups
      }
      print(
        f"{-F[i, 0]:8.3f} {F[i, 1]:10.2f} {n_motor_types:6d} {cum_tau:11.1f}  {design}"
      )
    rewards = -F[:, 0]
    costs = F[:, 1]
    max_reward = float(np.max(rewards))
    keep = rewards >= 0.97 * max_reward
    efficiency = np.full_like(costs, fill_value=-np.inf, dtype=np.float64)
    np.divide(rewards, costs, out=efficiency, where=costs > 0)
    candidate_idx = np.where(keep)[0]
    best_eff_idx = int(
      candidate_idx[np.argmax(efficiency[candidate_idx])] if len(candidate_idx) else 0
    )
    best_genome = (
      X[best_eff_idx] if isinstance(X[best_eff_idx], dict) else dict(X[best_eff_idx])
    )
    _print_final_solution_summary(
      title="efficiency pick from Pareto front",
      genome=best_genome,
      cfg=cfg,
      reward=float(rewards[best_eff_idx]),
      cost=float(costs[best_eff_idx]),
    )
    if out_path:
      np.savez(
        out_path,
        X=np.array([dict(x) for x in X], dtype=object),
        F=F,
        groups=[g.name for g in cfg.groups],
      )
      print(f"\nSaved results to {out_path}")
    return

  # For single-objective, sort by fitness (ascending = lower is better)
  order = np.argsort(F[:, 0])
  print("\n=== Best designs (sorted by fitness) ===")
  print(f"{'fitness':>10} {'motors':>7} {'cumTau(Nm)':>11}  design")
  rows = []
  for i in order:
    genome = X[i] if isinstance(X[i], dict) else dict(X[i])
    tau_vec, n_variants, cum_tau = decode_individual(genome, cfg)
    n_motor_types = count_motor_types(genome, cfg)
    fitness = F[i, 0]
    design = {
      g.name: round(float(tau_vec[cfg.joint_order.index(g.joints[0])]), 1)
      for g in cfg.groups
    }
    print(f"{fitness:10.3f} {n_motor_types:7d} {cum_tau:11.1f}  {design}")
    rows.append((fitness, n_motor_types, cum_tau, design))
  best_idx = int(order[0])
  best_genome = X[best_idx] if isinstance(X[best_idx], dict) else dict(X[best_idx])
  best_fitness = float(F[best_idx, 0])
  best_cost, _cum_tau, _n_types = _design_cost(best_genome, cfg)
  best_reward = best_cost - best_fitness
  _print_final_solution_summary(
    title="best scalarized fitness",
    genome=best_genome,
    cfg=cfg,
    reward=best_reward,
    cost=best_cost,
  )
  if out_path:
    np.savez(
      out_path,
      X=np.array([dict(x) for x in X], dtype=object),
      F=F,
      groups=[g.name for g in cfg.groups],
    )
    print(f"\nSaved results to {out_path}")


def _design_cost(genome: dict, cfg: CodesignConfig) -> tuple[float, float, int]:
  """Recompute (total_cost, cum_tau, n_types) for one genome under ``cfg`` weights.

  Mirrors the cost model in ``_make_problem._evaluate`` so that a winner's raw
  performance can be recovered from its scalarized fitness via
  ``perf = total_cost - fitness``.
  """
  tau_vec, n_choices, cum_tau = decode_individual(genome, cfg)
  n_types = count_motor_types(genome, cfg)
  base_cost = cfg.w_count * n_choices + cfg.w_torque * cfg.capacity_cost(
    tau_vec, cum_tau
  )
  excess_penalty = max(0.0, float(n_types - 3)) * cfg.w_motor_type_penalty
  return base_cost + excess_penalty, float(cum_tau), int(n_types)


def run_w_torque_sweep(
  backend: CodesignBackend,
  cfg: CodesignConfig,
  w_torque_values: list[float],
  pop_size: int,
  n_seeds: int,
  generations: int,
  seed: int,
  out_path: str | None,
) -> None:
  """Trace a reward-vs-cost Pareto front by sweeping ``w_torque``.

  The single-objective GA collapses reward and cost into one scalar, so a single
  run returns one design. Re-running it at a range of torque-cost weights walks
  the reward/cost trade-off: small ``w_torque`` favours high-torque high-reward
  designs, large ``w_torque`` favours cheap low-torque designs. Each run reuses
  the (expensive) Isaac Lab env. Winners are stored in the legacy multi-objective
  layout ``F = [[-performance, cum_tau], ...]`` so ``validate_codesign`` /
  ``visualize_codesign`` can select by reward or efficiency.
  """
  designs: list[dict] = []
  f_rows: list[list[float]] = []
  print(f"\n=== w_torque sweep over {w_torque_values} ===")
  for wt in w_torque_values:
    cfg.w_torque = wt
    print(f"\n--- w_torque = {wt} ---")
    result = run_optimization(
      backend, cfg, pop_size, n_seeds, generations, seed=seed, seed_genome=None
    )
    X = np.atleast_1d(result.X)
    F = np.atleast_2d(result.F)
    best = int(np.argmin(F[:, 0]))
    genome = X[best] if isinstance(X[best], dict) else dict(X[best])
    fitness = float(F[best, 0])
    total_cost, cum_tau, n_types = _design_cost(genome, cfg)
    perf = total_cost - fitness
    designs.append(dict(genome))
    f_rows.append([-perf, cum_tau])
    design = {
      g.name: round(
        float(decode_individual(genome, cfg)[0][cfg.joint_order.index(g.joints[0])]), 1
      )
      for g in cfg.groups
    }
    print(
      f"  winner: perf={perf:.3f} cum_tau={cum_tau:.1f}Nm "
      f"motor_types={n_types} design={design}"
    )

  F_arr = np.array(f_rows, dtype=np.float64)
  print("\n=== Sweep Pareto front (reward vs cum_tau) ===")
  print(f"{'w_torque':>9} {'perf':>10} {'cum_tau(Nm)':>12}")
  for wt, row in zip(w_torque_values, f_rows, strict=True):
    print(f"{wt:9.3f} {-row[0]:10.3f} {row[1]:12.1f}")
  if out_path:
    np.savez(
      out_path,
      X=np.array([dict(d) for d in designs], dtype=object),
      F=F_arr,
      groups=[g.name for g in cfg.groups],
      w_torque_values=np.array(w_torque_values, dtype=np.float64),
    )
    print(f"\nSaved sweep Pareto set to {out_path}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "--backend", choices=["mjlab", "isaaclab", "sim2sim", "wm"], default="mjlab"
  )
  p.add_argument(
    "--wm-checkpoint",
    default=None,
    help="Trained world-model checkpoint (wm-train output) for --backend wm.",
  )
  p.add_argument(
    "--wm-verify-topk",
    type=float,
    default=0.25,
    help="Fraction of each generation re-scored in true sim (mjlab backend) "
    "when using --backend wm; high-uncertainty designs are verified too. "
    "0 disables verification (pure surrogate).",
  )
  p.add_argument(
    "--robot",
    choices=["qdd", "gene"],
    default="qdd",
    help="Actuator design space + joint order. 'qdd' = QDD lower body (12 leg "
    "joints); 'gene' = Gene01 nowrist_noneck whole body (22 actuated joints).",
  )
  p.add_argument(
    "--task",
    default=None,
    help="Env/task id. Defaults per --robot (QDD MotorCond for qdd, gene01 "
    "nowrist_noneck motorcond play-rough for gene).",
  )
  p.add_argument(
    "--policy",
    default=None,
    help="Path to an exported TorchScript policy or a saved .pt checkpoint. "
    "Optional with --smoke-test (falls back to a random policy).",
  )
  p.add_argument("--pop-size", type=int, default=32)
  p.add_argument("--n-seeds", type=int, default=4, help="Rollout seeds per design.")
  p.add_argument("--rollout-steps", type=int, default=200)
  p.add_argument("--generations", type=int, default=25)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--w-count", type=float, default=1.0)
  p.add_argument("--w-torque", type=float, default=1.0)
  p.add_argument(
    "--cost-model",
    choices=["legacy", "linear", "powerlaw", "catalog"],
    default="legacy",
    help="Hardware-capacity cost model. 'legacy' uses the linear cum_tau/tau_ref "
    "proxy (original behavior). 'linear'/'powerlaw'/'catalog' use the physical "
    "actuator mass (kg) from codesign_motor_model; 'powerlaw' (mass~tau^alpha) is "
    "the grounded default for BLDC/QDD actuators.",
  )
  p.add_argument(
    "--cost-alpha",
    type=float,
    default=0.75,
    help="Power-law exponent for --cost-model powerlaw (0.7-0.8 for BLDC/QDD).",
  )
  p.add_argument(
    "--max-step-height",
    type=float,
    default=None,
    help="Cap the eval stairs terrain step height (m), e.g. 0.10, so the RMS/"
    "torque demand reflects the real deployment envelope instead of the training "
    "max (0.23 m). IsaacLab backend only; disables terrain curriculum.",
  )
  p.add_argument(
    "--rms-peak-ratio",
    type=float,
    default=0.25,
    help="Thermal actuator-sizing limit: per-joint RMS torque (from the rollout) "
    "must stay <= this fraction of the joint's peak torque (~0.25 = continuous "
    "rating of a BLDC/QDD motor). Enforced as a hard constraint (--multiobjective) "
    "or a strong penalty (single-objective). Set to 0 to disable.",
  )
  p.add_argument(
    "--w-torque-sweep",
    default=None,
    help="Comma-separated list of w_torque values (e.g. '0.25,0.5,1,2,4,8'). "
    "Runs the GA once per value, reusing the same env, and aggregates the "
    "per-value winners into a reward-vs-cost Pareto front. Overrides --w-torque.",
  )
  p.add_argument(
    "--multiobjective",
    action="store_true",
    help="Run a true bi-objective NSGA-II (minimize -reward and actuator mass) in "
    "ONE pass, returning the whole non-dominated front. No w_torque scalarization "
    "or sweep. Motor-type diversity is enforced as a <= --max-motor-types "
    "constraint. Overrides --w-torque-sweep.",
  )
  p.add_argument(
    "--max-motor-types",
    type=int,
    default=3,
    help="Distinct-actuator limit enforced as a constraint in --multiobjective mode.",
  )
  p.add_argument("--out", default="codesign_pareto.npz")
  p.add_argument(
    "--wandb",
    action="store_true",
    help="Log generation metrics + final Pareto "
    "front to Weights & Biases (requires prior `wandb login`).",
  )
  p.add_argument("--wandb-project", default="gene_codesign")
  p.add_argument(
    "--wandb-name",
    default=None,
    help="Run name. Defaults to an auto-generated name from the run config.",
  )
  # Performance objective (5): walkability score, task reward, or AMP alignment.
  p.add_argument("--objective", choices=["reward", "amp", "walk"], default="walk")
  p.add_argument(
    "--discriminator",
    default=None,
    help="Path to exported TorchScript AMP discriminator (required for --objective amp).",
  )
  p.add_argument(
    "--amp-transition",
    action="store_true",
    help="Feed the discriminator a concatenated (prev, cur) AMP-obs transition.",
  )
  # Quick end-to-end plumbing check: tiny run, random policy if none provided.
  p.add_argument(
    "--smoke-test",
    action="store_true",
    help="Tiny run (pop=4, seeds=1, steps=5, gens=1) to validate the pipeline.",
  )
  args = p.parse_args()

  # Resolve the robot-specific design space, default task and tau range.
  if args.robot == "gene":
    groups, joint_order, tau_obs_range = (
      GENE_GROUPS,
      GENE_JOINT_ORDER,
      GENE_TAU_OBS_RANGE,
    )
    default_task = GENE_TASK_DEFAULT
    # The walkability metric relies on the QDD-only sim2sim MuJoCo-C rollout; the
    # gene design space is evaluated in Isaac Lab against task reward instead.
    if args.objective == "walk":
      print("[Config] robot=gene: 'walk' objective unsupported; using 'reward'.")
      args.objective = "reward"
    if args.backend == "mjlab":
      print("[Config] robot=gene: switching backend to isaaclab.")
      args.backend = "isaaclab"
  else:
    groups, joint_order, tau_obs_range = QDD_GROUPS, QDD_JOINT_ORDER, TAU_OBS_RANGE
    default_task = "Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond"
  if args.task is None:
    args.task = default_task

  if args.backend == "wm" and args.objective != "reward":
    # The world model's reward head predicts task reward; walk/amp metrics
    # need a live simulator rollout.
    print(
      f"[Config] backend=wm: objective '{args.objective}' unsupported; using 'reward'."
    )
    args.objective = "reward"

  if args.objective == "walk" and args.backend != "sim2sim":
    print("[Config] Switching to sim2sim backend for walk objective.")
    args.backend = "sim2sim"

  if args.smoke_test:
    args.pop_size, args.n_seeds, args.rollout_steps, args.generations = 4, 1, 5, 1
    print("[smoke-test] pop=4 seeds=1 steps=5 gens=1")
    if args.policy is None:
      print("[smoke-test] no --policy given: using a random policy.")
  elif args.policy is None:
    p.error("--policy is required unless --smoke-test is set.")

  cost_model = (
    None
    if args.cost_model == "legacy"
    else MotorCostModel(kind=args.cost_model, alpha=args.cost_alpha)
  )
  cfg = CodesignConfig(
    groups=groups,
    joint_order=joint_order,
    tau_obs_range=tau_obs_range,
    w_count=args.w_count,
    w_torque=args.w_torque,
    cost_model=cost_model,
    rms_peak_ratio=args.rms_peak_ratio,
  )
  if args.rms_peak_ratio > 0:
    print(
      f"[Config] RMS thermal constraint ON: RMS torque <= "
      f"{args.rms_peak_ratio:.2f} * peak torque per joint "
      f"({'hard constraint' if args.multiobjective else 'strong penalty'})."
    )
  if cost_model is not None:
    print(
      f"[Config] cost-model={args.cost_model}"
      + (f" (alpha={args.cost_alpha})" if args.cost_model == "powerlaw" else "")
      + " -> capacity term is total actuator mass (kg)."
    )

  if args.wandb:
    import wandb

    run_name = args.wandb_name or (
      f"{args.robot}_{'mo' if args.multiobjective else 'so'}"
      f"_types{args.max_motor_types}_{args.cost_model}"
      f"_pop{args.pop_size}_gen{args.generations}"
    )
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

  # Build the performance metric (objective 5).
  metric: PerformanceMetric
  if args.objective == "amp":
    if args.discriminator is None:
      p.error("--discriminator is required for --objective amp.")
    disc = torch.jit.load(args.discriminator, map_location=args.device).eval()
    metric = AmpAlignmentMetric(disc, use_transition=args.amp_transition)
  elif args.objective == "walk":
    metric = WalkabilityMetric()
  else:
    metric = RewardMetric()

  # The env batch must hold the whole population times the per-design seeds.
  num_envs = args.pop_size * args.n_seeds

  # Isaac Lab requires the Omniverse Kit app to be launched *before* any isaaclab
  # import or env creation. Do it here (headless) so the IsaacLabBackend can build
  # the motor-conditioned gene env. Keep a handle to close it cleanly at the end.
  simulation_app = None
  if args.backend == "isaaclab":
    from isaaclab.app import AppLauncher  # type: ignore

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    import gb_rl_locomotion.networks  # type: ignore # noqa: F401
    import gb_rl_locomotion.tasks  # type: ignore # noqa: F401  (registers gym tasks)

  backend_kwargs = dict(
    task=args.task,
    policy_path=args.policy,
    num_envs=num_envs,
    joint_order=cfg.joint_order,
    tau_obs_range=cfg.tau_obs_range,
    rollout_steps=args.rollout_steps,
    metric=metric,
    device=args.device,
  )
  if args.backend == "isaaclab":
    backend_kwargs["max_step_height"] = args.max_step_height
  if args.backend == "wm":
    backend_kwargs["wm_checkpoint"] = args.wm_checkpoint
    backend_kwargs["wm_verify_topk"] = args.wm_verify_topk
  backend = build_backend(args.backend, **backend_kwargs)

  exit_code = 0
  try:
    if args.multiobjective:
      result = run_optimization(
        backend,
        cfg,
        pop_size=args.pop_size,
        n_seeds=args.n_seeds,
        generations=args.generations,
        seed=args.seed,
        seed_genome=WALK_SEED_GENOME if args.objective == "walk" else None,
        multiobjective=True,
        max_motor_types=args.max_motor_types,
      )
      report_pareto(result, cfg, args.out)
    elif args.w_torque_sweep is not None:
      sweep_values = [float(v) for v in args.w_torque_sweep.split(",") if v.strip()]
      run_w_torque_sweep(
        backend,
        cfg,
        sweep_values,
        pop_size=args.pop_size,
        n_seeds=args.n_seeds,
        generations=args.generations,
        seed=args.seed,
        out_path=args.out,
      )
    else:
      result = run_optimization(
        backend,
        cfg,
        pop_size=args.pop_size,
        n_seeds=args.n_seeds,
        generations=args.generations,
        seed=args.seed,
        seed_genome=WALK_SEED_GENOME if args.objective == "walk" else None,
      )
      report_pareto(result, cfg, args.out)

    if args.wandb:
      try:
        import wandb

        f = result.F if result.F.ndim == 2 else result.F.reshape(-1, 1)
        if f.shape[1] == 2:
          table = wandb.Table(
            columns=["reward", "cost"],
            data=[[float(-row[0]), float(row[1])] for row in f],
          )
          wandb.log({"final_pareto_front": table})
        wandb.finish()
      except Exception as exc:  # noqa: BLE001
        print(f"[wandb] final logging failed (non-fatal): {exc!r}")
  except Exception:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    exit_code = 1
  finally:
    if simulation_app is not None:
      # Isaac Sim's simulation_app.close() reliably hangs in a shutdown spin-loop
      # (_app_control_on_stop_handle_fn -> render -> cuda.set_device). All results
      # (npz) and wandb are already flushed above, so bypass the broken teardown
      # and hard-exit so a sweep loop can advance to the next run.
      # ponytail: os._exit skips atexit/Isaac cleanup; safe, outputs already persisted
      sys.stdout.flush()
      sys.stderr.flush()
      os._exit(exit_code)


if __name__ == "__main__":
  main()
