"""Differentiable motor-type assignment via Gumbel-Softmax.

This module implements a learnable actuator wrapper that jointly optimizes:
1. Per-joint assignment to one of K motor types (discrete → relaxed)
2. Per-type torque limits τ_max[k]

Architecture: alternating optimization between PPO (policy) and a separate
codesign optimizer (alpha, tau_raw). The codesign step uses a differentiable
surrogate computed from logged torques — not through the simulator.

Key idea: instead of hard `clip(τ, -τ_max, τ_max)`, use
  τ_applied = τ_max_eff × tanh(τ_cmd / τ_max_eff)
which is smooth and differentiable everywhere.

Training protocol:
  1. Rollout: actuator uses deterministic soft assignment (no Gumbel noise)
  2. PPO update: policy learns under current torque limits
  3. Codesign step: separate optimizer updates alpha/tau_raw via surrogate loss
  4. Temperature anneals to harden assignment toward discrete
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SymmetrySpec:
  """Defines left/right symmetry for joint assignment sharing.

  Joints in the same symmetry group share assignment logits, ensuring
  e.g. l_hip_pitch and r_hip_pitch always get the same motor type.
  """

  joint_names: list[str]
  """All joint names (full list, including both sides)."""

  symmetry_pairs: list[tuple[str, str]]
  """Pairs of (left, right) joints that share parameters."""

  @property
  def unique_joints(self) -> list[str]:
    """Unique joints after symmetry collapse (keeps first of each pair)."""
    right_set = {pair[1] for pair in self.symmetry_pairs}
    return [j for j in self.joint_names if j not in right_set]

  @property
  def n_unique(self) -> int:
    return len(self.unique_joints)

  def unique_index_for(self, joint_name: str) -> int:
    """Map any joint name to its unique index (symmetric joints → same idx)."""
    unique = self.unique_joints
    for left, right in self.symmetry_pairs:
      if joint_name == right:
        return unique.index(left)
    return unique.index(joint_name)

  def expand_to_full(self, unique_values: torch.Tensor) -> torch.Tensor:
    """Expand unique-joint tensor to full joint-order tensor.

    Args:
      unique_values: Shape (..., n_unique) or (..., n_unique, K)

    Returns:
      Tensor with shape (..., n_joints) or (..., n_joints, K)
    """
    indices = [self.unique_index_for(j) for j in self.joint_names]
    idx_tensor = torch.tensor(indices, device=unique_values.device)
    if unique_values.dim() >= 2 and unique_values.shape[-1] != len(self.joint_names):
      return unique_values[..., idx_tensor, :]
    return unique_values[..., idx_tensor]


@dataclass
class CodesignConfig:
  """Configuration for the Gumbel-Softmax codesign module."""

  n_types: int = 2
  """Number of motor types to learn."""

  init_tau_max: list[float] = field(default_factory=lambda: [80.0, 40.0])
  """Initial τ_max per type (Nm). Length must equal n_types."""

  temperature_init: float = 1.0
  """Initial Gumbel-Softmax temperature."""

  temperature_min: float = 0.1
  """Minimum temperature after annealing."""

  temperature_decay: float = 0.9995
  """Multiplicative decay applied each iteration."""

  lambda_types: float = 0.01
  """Weight for type-count loss (fewer types = better)."""

  lambda_balance: float = 0.05
  """Weight for balance loss (prevent outlier assignments)."""

  lambda_tau: float = 1e-4
  """Weight for downward pressure on τ_max values."""

  lambda_saturation: float = 0.1
  """Weight for soft saturation-risk loss in surrogate."""

  lambda_rms: float = 0.05
  """Weight for surrogate RMS torque loss."""

  lambda_peak: float = 0.1
  """Weight for surrogate peak torque loss."""

  min_tau: float = 5.0
  """Minimum τ_max in Nm (physical lower bound)."""

  max_tau: float = 120.0
  """Maximum τ_max in Nm (physical upper bound)."""

  codesign_lr: float = 3e-3
  """Learning rate for the codesign optimizer."""

  codesign_interval: int = 5
  """Update codesign params every N PPO iterations."""


class GumbelSoftmaxActuator(nn.Module):
  """Differentiable motor-type assignment with alternating optimization.

  During rollout (inference mode):
    - Uses deterministic softmax assignment (no Gumbel noise)
    - Applies soft saturation: τ_max_eff × tanh(τ_cmd / τ_max_eff)
    - Logs commanded torques for the codesign step

  During codesign step (gradient mode):
    - Recomputes soft saturation differentiably from logged torques
    - Optimizes alpha/tau_raw via surrogate losses
    - Uses Gumbel noise for exploration in assignment space

  Learnable parameters:
    - alpha: (n_unique, K) assignment logits
    - tau_raw: (K,) per-type torque limits (sigmoid-bounded)
  """

  def __init__(
    self,
    cfg: CodesignConfig,
    symmetry: SymmetrySpec,
  ) -> None:
    super().__init__()
    self.cfg = cfg
    self.symmetry = symmetry

    n_unique = symmetry.n_unique
    K = cfg.n_types

    assert len(cfg.init_tau_max) == K, (
      f"init_tau_max length ({len(cfg.init_tau_max)}) must equal n_types ({K})"
    )

    # Assignment logits — initialized near-uniform with small noise.
    self.alpha = nn.Parameter(torch.randn(n_unique, K) * 0.1)

    # Per-type tau_max via sigmoid: τ = min + (max - min) * σ(raw).
    init_raw = torch.tensor(
      [_inverse_sigmoid_bounded(t, cfg.min_tau, cfg.max_tau) for t in cfg.init_tau_max],
      dtype=torch.float,
    )
    self.tau_raw = nn.Parameter(init_raw)

    # Temperature state (not a parameter — managed externally).
    self.register_buffer(
      "temperature", torch.tensor(cfg.temperature_init, dtype=torch.float)
    )

    # Torque log buffer for codesign step (detached rollout data).
    self._torque_log: list[torch.Tensor] = []

  @property
  def tau_max(self) -> torch.Tensor:
    """Per-type torque limits (Nm), shape (K,). Bounded via sigmoid."""
    return self.cfg.min_tau + (self.cfg.max_tau - self.cfg.min_tau) * torch.sigmoid(
      self.tau_raw
    )

  def tau_eff(self, use_gumbel: bool = False) -> torch.Tensor:
    """Effective per-joint torque limit, shape (n_joints,).

    Args:
      use_gumbel: If True, use Gumbel-Softmax noise (for codesign step).
                  If False, use deterministic softmax (for rollout).
    """
    if use_gumbel and self.training:
      p = F.gumbel_softmax(self.alpha, tau=self.temperature.item(), hard=False)
    else:
      p = F.softmax(self.alpha / max(self.temperature.item(), 0.01), dim=-1)

    tau_types = self.tau_max  # (K,)
    tau_eff_unique = (p * tau_types.unsqueeze(0)).sum(dim=-1)  # (n_unique,)
    return self.symmetry.expand_to_full(tau_eff_unique)  # (n_joints,)

  def forward(self, tau_command: torch.Tensor) -> torch.Tensor:
    """Apply soft saturation during rollout (deterministic, no Gumbel).

    Args:
      tau_command: Commanded torques, shape (num_envs, n_joints).

    Returns:
      Saturated torques, same shape.
    """
    with torch.no_grad():
      tau_eff = self.tau_eff(use_gumbel=False)  # (n_joints,)
      tau_eff_expanded = tau_eff.unsqueeze(0)  # (1, n_joints)
      ratio = tau_command / (tau_eff_expanded + 1e-6)
      saturated = tau_eff_expanded * torch.tanh(ratio)
    return saturated

  def log_torques(self, tau_command: torch.Tensor) -> None:
    """Log commanded torques for the codesign optimizer step.

    Call this with the PD-computed torques (before saturation) during rollout.
    Tensors are detached and moved to CPU to save GPU memory.
    """
    self._torque_log.append(tau_command.detach().cpu())

  def clear_torque_log(self) -> None:
    """Clear logged torques after codesign step."""
    self._torque_log.clear()

  def surrogate_loss(self, device: str | torch.device = "cpu") -> torch.Tensor:
    """Compute differentiable surrogate loss from logged torques.

    This is the main loss for the codesign optimizer. It re-applies soft
    saturation differentiably to logged (detached) torque commands, then
    penalizes:
      1. RMS of applied torques (normalized by fixed reference)
      2. Peak (max) applied torque across joints
      3. Soft saturation risk: how close joints are to their limits
      4. Structural codesign regularizers (type count, balance, tau pressure)

    Returns:
      Scalar loss for backprop through alpha and tau_raw.
    """
    if not self._torque_log:
      return torch.tensor(0.0, device=device, requires_grad=True)

    # Stack logged torques: (N_total, n_joints)
    tau_logged = torch.cat(self._torque_log, dim=0).to(device)

    # Differentiable tau_eff with Gumbel noise for exploration.
    tau_eff = self.tau_eff(use_gumbel=True)  # (n_joints,)
    tau_eff_expanded = tau_eff.unsqueeze(0)  # (1, n_joints)

    # Re-apply soft saturation differentiably.
    ratio = tau_logged / (tau_eff_expanded + 1e-6)
    tau_applied = tau_eff_expanded * torch.tanh(ratio)

    # Reference normalization (fixed, not learned — prevents gaming).
    tau_ref = self.cfg.max_tau

    # --- RMS penalty: penalize mean squared applied torque ---
    rms_per_env = torch.sqrt(torch.mean(tau_applied**2, dim=1) + 1e-8)
    l_rms = self.cfg.lambda_rms * (rms_per_env / tau_ref).mean()

    # --- Peak penalty: penalize maximum absolute torque ---
    peak_per_env = torch.max(torch.abs(tau_applied), dim=1).values
    l_peak = self.cfg.lambda_peak * (peak_per_env / tau_ref).mean()

    # --- Soft saturation risk: sigmoid proximity to limit ---
    sat_ratio = torch.abs(tau_applied) / (tau_eff_expanded + 1e-6)
    sat_risk = torch.sigmoid(20.0 * (sat_ratio - 0.8))
    l_sat = self.cfg.lambda_saturation * sat_risk.mean()

    # --- Structural losses (type count, balance, tau pressure) ---
    l_structural = self._structural_loss()

    return l_rms + l_peak + l_sat + l_structural

  def step_temperature(self) -> None:
    """Decay temperature by one step. Call once per training iteration."""
    new_temp = self.temperature * self.cfg.temperature_decay
    self.temperature.fill_(max(new_temp.item(), self.cfg.temperature_min))

  def hard_assignment(self) -> dict[str, int]:
    """Get discrete assignment: argmax over alpha logits."""
    type_indices = self.alpha.argmax(dim=-1)
    result: dict[str, int] = {}
    for j_name in self.symmetry.joint_names:
      u_idx = self.symmetry.unique_index_for(j_name)
      result[j_name] = int(type_indices[u_idx].item())
    return result

  def hard_tau_max(self) -> list[float]:
    """Get final per-type τ_max values (Nm)."""
    return self.tau_max.detach().tolist()

  def _structural_loss(self) -> torch.Tensor:
    """Combined structural codesign regularization loss."""
    p = F.softmax(self.alpha / max(self.temperature.item(), 0.01), dim=-1)

    # Type count: penalize having many active types.
    usage = p.mean(dim=0)
    active = torch.sigmoid(10.0 * (usage - 0.1))
    l_types = self.cfg.lambda_types * active.sum()

    # Balance: penalize one type dominating (> 70% usage).
    max_usage = usage.max()
    excess = F.relu(max_usage - 0.7)
    l_balance = self.cfg.lambda_balance * excess**2

    # Tau pressure: encourage smaller motors.
    l_tau = self.cfg.lambda_tau * self.tau_max.sum()

    return l_types + l_balance + l_tau

  def summary(self) -> str:
    """Human-readable summary of current assignment state."""
    assignment = self.hard_assignment()
    tau_values = self.hard_tau_max()
    lines = [
      f"Temperature: {self.temperature.item():.4f}",
      f"τ_max per type: {[f'{t:.1f} Nm' for t in tau_values]}",
      "Assignment:",
    ]
    from collections import defaultdict

    groups: dict[int, list[str]] = defaultdict(list)
    for joint, t_idx in assignment.items():
      groups[t_idx].append(joint)
    for t_idx in sorted(groups):
      joints = groups[t_idx]
      lines.append(
        f"  Type {t_idx} (τ={tau_values[t_idx]:.1f} Nm): "
        f"{len(joints)} joints — {joints[:5]}{'...' if len(joints) > 5 else ''}"
      )
    return "\n".join(lines)


class CodesignScheduler:
  """Alternating optimizer for codesign parameters.

  Manages the separate optimization of alpha/tau_raw alongside PPO training.
  Updates codesign params every `codesign_interval` PPO iterations using
  surrogate losses computed from logged torques.

  Usage in training loop:
    scheduler = CodesignScheduler(gumbel_actuator)
    for ppo_iter in range(num_iters):
        # ... rollout with actuator.forward() ...
        # ... PPO update ...
        scheduler.step(ppo_iter)
  """

  def __init__(
    self,
    actuator: GumbelSoftmaxActuator,
    device: str | torch.device = "cuda",
  ) -> None:
    self.actuator = actuator
    self.device = device
    self._optimizer = torch.optim.Adam(
      actuator.parameters(), lr=actuator.cfg.codesign_lr
    )
    self._iteration = 0

  def step(self, ppo_iter: int) -> dict[str, float] | None:
    """Perform codesign update if at the right interval.

    Args:
      ppo_iter: Current PPO iteration number.

    Returns:
      Dict of loss components if update was performed, None otherwise.
    """
    self._iteration = ppo_iter

    # Always anneal temperature.
    self.actuator.step_temperature()

    # Only update codesign params every N iterations.
    if ppo_iter % self.actuator.cfg.codesign_interval != 0:
      return None

    if not self.actuator._torque_log:
      return None

    # Enable gradients for codesign step.
    self.actuator.train()
    self._optimizer.zero_grad()

    loss = self.actuator.surrogate_loss(device=self.device)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.actuator.parameters(), max_norm=1.0)
    self._optimizer.step()

    # Switch back to eval for rollout.
    self.actuator.eval()
    self.actuator.clear_torque_log()

    with torch.no_grad():
      return {
        "codesign/total_loss": loss.item(),
        "codesign/temperature": self.actuator.temperature.item(),
        "codesign/tau_max": self.actuator.tau_max.detach().cpu().tolist(),
        "codesign/n_types_active": int(
          (F.softmax(self.actuator.alpha, dim=-1).mean(dim=0) > 0.1).sum().item()
        ),
      }


def _inverse_sigmoid_bounded(y: float, lo: float, hi: float) -> float:
  """Compute raw value such that lo + (hi - lo) * sigmoid(raw) ≈ y."""
  t = (y - lo) / (hi - lo)
  t = max(1e-6, min(1.0 - 1e-6, t))
  return math.log(t / (1.0 - t))
