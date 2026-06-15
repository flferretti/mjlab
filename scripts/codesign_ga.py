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
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

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
QDD_GROUPS: list[ActuatorGroup] = [
  ActuatorGroup("hip_pitch", ("l_hip_pitch", "r_hip_pitch"), 80.0, (30.0, 150.0)),
  ActuatorGroup("hip_roll", ("l_hip_roll", "r_hip_roll"), 80.0, (30.0, 150.0)),
  ActuatorGroup("hip_yaw", ("l_hip_yaw", "r_hip_yaw"), 40.0, (10.0, 100.0)),
  ActuatorGroup("knee", ("l_knee", "r_knee"), 80.0, (30.0, 180.0)),  # Increased to Type 3 capacity
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


# ---------------------------------------------------------------------------
# 2. Genome <-> design decoding.
# ---------------------------------------------------------------------------
# Pure continuous genome, per group g:
#   tau_g in [lo_g, hi_g]: max torque limit (Nm) for joint group g.
# (No discrete custom flag; all variables are continuous for richer search space)


def group_var_names(group: ActuatorGroup) -> tuple[str]:
  """Return only the torque variable name (no custom flag)."""
  return (f"tau_{group.name}",)


def count_motor_types(genome: dict[str, float], cfg: CodesignConfig) -> int:
  """Count distinct motor types using hierarchical clustering.

  Uses hierarchical agglomerative clustering with a distance threshold to adaptively
  group similar torques into the same motor type. This is more flexible than fixed
  rounding because it:
    1. Groups similar values regardless of their absolute values
    2. Counts "natural" clusters rather than arbitrary thresholds
    3. Scales better across the full [5, 90] Nm range

  Distance threshold: 2.0 Nm means torques within ±2.0 Nm of each other cluster.

  Examples:
    τ = [80.1, 79.9, 51.0, 49.5, 25.0]
    → With threshold 2.0:
       * 80.1, 79.9 cluster together (distance 0.2 < 2.0) → Type 1
       * 51.0, 49.5 cluster together (distance 1.5 < 2.0) → Type 2
       * 25.0 alone → Type 3
    → count: 3 types (natural clustering!)

    τ = [80.0, 70.0, 50.0, 40.0, 30.0, 20.0]
    → All inter-group distances > 2.0, so no merging
    → count: 6 types

  Algorithm:
    1. Build hierarchical clustering dendrogram of 6 tau values
    2. Cut dendrogram at distance threshold (2.0 Nm)
    3. Count resulting clusters
  """
  taus = np.array([float(genome[f"tau_{group.name}"]) for group in cfg.groups]).reshape(
    -1, 1
  )  # Reshape for scipy

  # Hierarchical clustering (single linkage is sensitive, complete is robust)
  if len(taus) <= 1:
    return len(taus)

  # Compute pairwise distances and build dendrogram
  distances = pdist(taus, metric="euclidean")
  linkage_matrix = linkage(distances, method="complete")

  # Cut at threshold: 2.0 Nm means "same type if within ±2.0"
  # (Complete linkage: max distance between any pair in cluster)
  cluster_labels = fcluster(linkage_matrix, t=2.0, criterion="distance")

  return len(np.unique(cluster_labels))


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

  Handles dimension mismatch between IsaacSim training (177 obs) and mjlab (168 obs)
  by zero-padding or truncating observations as needed.

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

    # ONNX expects [1, onnx_obs_dim], mjlab provides [batch, 168]
    # Extract expected obs dimension from ONNX model
    self._onnx_obs_dim = int(self._input_shape[1])
    print(
      f"[OnnxPolicy] Loaded {onnx_path}: expects obs_dim={self._onnx_obs_dim}, "
      f"batch_size={self._input_shape[0]}"
    )

  def _adapt_obs(self, obs: torch.Tensor) -> np.ndarray:
    """Convert mjlab obs (batch, 168) to ONNX obs (batch, onnx_obs_dim).

    If mjlab obs_dim < onnx_obs_dim: pad with zeros.
    If mjlab obs_dim > onnx_obs_dim: truncate.
    """
    mjlab_obs_dim = obs.shape[-1]
    if mjlab_obs_dim == self._onnx_obs_dim:
      return obs.cpu().numpy().astype(np.float32)

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
      action = self._sess.run([self._output_name], {self._input_name: obs_single})[0]
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

  @abc.abstractmethod
  def _reset(self) -> dict: ...

  @abc.abstractmethod
  def _step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor]: ...

  @abc.abstractmethod
  def _actor_obs(self, obs: dict) -> torch.Tensor: ...

  @abc.abstractmethod
  def _policy(self, actor_obs: torch.Tensor) -> torch.Tensor: ...

  @abc.abstractmethod
  def _write_tau(self, tau_full: torch.Tensor) -> None:
    """Write the (num_envs, n_joints) torque matrix to obs buffer + effort limits."""

  @abc.abstractmethod
  def action_dim(self) -> int:
    """Total action dimension (for the random smoke-test policy)."""

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
    for _ in range(self.rollout_steps):
      actions = self._policy(self._actor_obs(obs))
      obs, reward = self._step(actions)
      self.metric.update(self, obs, reward)
      # Re-assert each step so any auto-reset on termination keeps the design.
      self._write_tau(self._tau_full)
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

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    # Disable the training-time torque randomization so the GA controls tau_max.
    if getattr(cfg, "events", None) is not None:
      cfg.events.pop("randomize_motor_tau_max", None)
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

  def _write_tau(self, tau_full: torch.Tensor) -> None:
    self._set_tau(
      self.env.unwrapped,
      self._env_ids,
      tau_full,
      self.joint_order,
      self._tau_obs_range,
    )


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
    self.env = gym.make(task, cfg=env_cfg)

    if policy_path is not None:
      self.policy_module = torch.jit.load(policy_path, map_location=device).eval()
    else:
      self.policy_module = RandomPolicy(self.action_dim(), device)

    from gb_rl_locomotion.mdp.motor_randomization import (  # type: ignore
      set_motor_tau_max,
    )

    self._set_tau = set_motor_tau_max
    self._env_ids = torch.arange(num_envs, device=device)

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

  def _write_tau(self, tau_full: torch.Tensor) -> None:
    self._set_tau(
      self.env.unwrapped,
      self._env_ids,
      tau_full,
      self.joint_order,
      self._tau_obs_range,
    )


def build_backend(name: str, **kwargs) -> CodesignBackend:
  """Factory: switch simulation backend with a single argument."""
  if name == "mjlab":
    return MjlabBackend(**kwargs)
  if name == "isaaclab":
    return IsaacLabBackend(**kwargs)
  raise ValueError(f"Unknown backend '{name}' (expected 'mjlab' or 'isaaclab').")


# ---------------------------------------------------------------------------
# 4. pymoo mixed-variable, multi-objective problem (vectorized GPU evaluation).
# ---------------------------------------------------------------------------


def _make_problem(backend: CodesignBackend, cfg: CodesignConfig, n_seeds: int):
  from pymoo.core.problem import Problem
  from pymoo.core.variable import Real

  # Pure continuous variables: per group, optimize torque limit (Nm)
  # Ordered dict preserves variable order for encoding/decoding
  from collections import OrderedDict

  variables = OrderedDict()
  for group in cfg.groups:
    (tau_name,) = group_var_names(group)
    variables[tau_name] = Real(bounds=group.tau_bounds)

  class CodesignProblem(Problem):
    def __init__(self) -> None:
      # n_obj=2: [-performance, hardware_cost]
      # No hard constraint; soft penalty for >3 motor types via cost function
      super().__init__(vars=variables, n_obj=2)

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

      f_perf = -performance  # maximize reward -> minimize negative reward

      # Cost function with soft penalty for excess motor types
      f_cost = []
      for (n_choices, cum_tau), n_types in zip(costs, n_motor_types):
        # Base cost: diversity (number of unique motor models) + total capacity
        base_cost = cfg.w_count * n_choices + cfg.w_torque * (cum_tau / cfg.tau_ref)

        # STRONG penalty: strongly discourage >3 types so Pareto front naturally
        # evolves toward 2-3 types. Multiplier must exceed typical reward variance.
        # E.g., if reward ranges ~0.05-0.1 and cost ~5-15, penalty must be ~10+ to
        # make "4 types" designs Pareto-dominated by "3 types" designs.
        excess_penalty = max(0.0, float(n_types - 3)) * cfg.w_motor_type_penalty
        f_cost.append(base_cost + excess_penalty)

      out["F"] = np.column_stack([f_perf, np.array(f_cost, dtype=np.float64)])

  return CodesignProblem()


def run_optimization(
  backend: CodesignBackend,
  cfg: CodesignConfig,
  pop_size: int,
  n_seeds: int,
  generations: int,
  seed: int = 0,
):
  from pymoo.algorithms.moo.nsga2 import NSGA2
  from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
  )
  from pymoo.optimize import minimize

  problem = _make_problem(backend, cfg, n_seeds)

  # NSGA-II with MixedVariableSampling to handle dict-based continuous variables
  # MixedVariableSampling works for pure continuous dict variables too
  algorithm = NSGA2(
    pop_size=pop_size,
    n_offsprings=pop_size,
    sampling=MixedVariableSampling(),
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
  X = np.atleast_1d(result.X)
  F = np.atleast_2d(result.F)

  # Validate result shape
  if F.ndim != 2 or F.shape[1] < 2:
    print(f"\n⚠️  Unexpected result shape: F.shape={F.shape}")
    print(f"Expected (n_designs, 2) objectives, got {F.shape}")
    return

  order = np.argsort(F[:, 1])  # by ascending hardware cost
  print("\n=== Pareto front (performance vs hardware cost) ===")
  print(f"{'reward':>10} {'cost':>10} {'motors':>7} {'cumTau(Nm)':>11}  design")
  rows = []
  for i in order:
    genome = X[i] if isinstance(X[i], dict) else dict(X[i])
    tau_vec, n_variants, cum_tau = decode_individual(genome, cfg)
    n_motor_types = count_motor_types(genome, cfg)
    reward = -F[i, 0]
    cost = F[i, 1]
    design = {
      g.name: round(float(tau_vec[cfg.joint_order.index(g.joints[0])]), 1)
      for g in cfg.groups
    }
    print(f"{reward:10.3f} {cost:10.3f} {n_motor_types:7d} {cum_tau:11.1f}  {design}")
    rows.append((reward, cost, n_motor_types, cum_tau, design))
  if out_path:
    np.savez(
      out_path,
      X=np.array([dict(x) for x in X], dtype=object),
      F=F,
      groups=[g.name for g in cfg.groups],
    )
    print(f"\nSaved Pareto set to {out_path}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backend", choices=["mjlab", "isaaclab"], default="mjlab")
  p.add_argument("--task", default="Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond")
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
  p.add_argument("--out", default="codesign_pareto.npz")
  # Performance objective (5): task reward, or AMP-discriminator alignment.
  p.add_argument("--objective", choices=["reward", "amp"], default="reward")
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

  if args.smoke_test:
    args.pop_size, args.n_seeds, args.rollout_steps, args.generations = 4, 1, 5, 1
    print("[smoke-test] pop=4 seeds=1 steps=5 gens=1")
    if args.policy is None:
      print("[smoke-test] no --policy given: using a random policy.")
  elif args.policy is None:
    p.error("--policy is required unless --smoke-test is set.")

  cfg = CodesignConfig(
    groups=QDD_GROUPS,
    joint_order=QDD_JOINT_ORDER,
    tau_obs_range=TAU_OBS_RANGE,
    w_count=args.w_count,
    w_torque=args.w_torque,
  )

  # Build the performance metric (objective 5).
  metric: PerformanceMetric
  if args.objective == "amp":
    if args.discriminator is None:
      p.error("--discriminator is required for --objective amp.")
    disc = torch.jit.load(args.discriminator, map_location=args.device).eval()
    metric = AmpAlignmentMetric(disc, use_transition=args.amp_transition)
  else:
    metric = RewardMetric()

  # The env batch must hold the whole population times the per-design seeds.
  num_envs = args.pop_size * args.n_seeds
  backend = build_backend(
    args.backend,
    task=args.task,
    policy_path=args.policy,
    num_envs=num_envs,
    joint_order=cfg.joint_order,
    tau_obs_range=cfg.tau_obs_range,
    rollout_steps=args.rollout_steps,
    metric=metric,
    device=args.device,
  )

  result = run_optimization(
    backend,
    cfg,
    pop_size=args.pop_size,
    n_seeds=args.n_seeds,
    generations=args.generations,
    seed=args.seed,
  )
  report_pareto(result, cfg, args.out)


if __name__ == "__main__":
  main()
