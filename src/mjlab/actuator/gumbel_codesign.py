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

  def __post_init__(self) -> None:
    names = set(self.joint_names)
    seen_right: set[str] = set()
    for left, right in self.symmetry_pairs:
      if left not in names:
        raise ValueError(f"Symmetry pair left joint '{left}' not in joint_names.")
      if right not in names:
        raise ValueError(f"Symmetry pair right joint '{right}' not in joint_names.")
      if right in seen_right:
        raise ValueError(f"Joint '{right}' appears as the right side of two pairs.")
      seen_right.add(right)

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

  def reduce_to_unique_max(self, full_values: torch.Tensor) -> torch.Tensor:
    """Reduce a full per-joint tensor to unique joints via max over each group.

    Symmetric L/R joints map to the same unique index; the max is taken so a
    shared motor covers the more demanding side.

    Args:
      full_values: Shape (n_joints,).

    Returns:
      Tensor of shape (n_unique,).
    """
    indices = [self.unique_index_for(j) for j in self.joint_names]
    idx_tensor = torch.tensor(indices, device=full_values.device)
    out = torch.zeros(self.n_unique, device=full_values.device, dtype=full_values.dtype)
    out.scatter_reduce_(0, idx_tensor, full_values, reduce="amax", include_self=False)
    return out


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

  # --- Demand-driven clustering parameters ---

  lambda_deficit: float = 2.0
  """Weight for the covering deficit loss. Drives τ_max UP to cover the torque
  demand of assigned joints. Linear (L1); must dominate the assignment cost so
  every joint stays covered (keep lambda_deficit > lambda_motor_cost)."""

  lambda_motor_cost: float = 0.5
  """Weight for the per-joint assignment cost. Drives each joint toward the
  smallest motor that still covers it, producing tiered clustering. Too small
  → motors bloat; too large (≥ lambda_deficit) → undersizing."""

  demand_quantile: float = 0.99
  """Quantile of |unclipped torque| used as the per-joint demand. High quantile
  (not max) for robustness to transient spikes, but high enough to cover peaks."""

  demand_margin: float = 1.1
  """Safety factor applied to the demand quantile when sizing motors."""

  usage_threshold: float = 0.05
  """Usage fraction below which a motor type is considered inactive (pruned)."""

  usage_sharpness: float = 20.0
  """Sharpness of the soft active-type indicator sigmoid."""

  min_tau: float = 5.0
  """Minimum τ_max in Nm (physical lower bound)."""

  max_tau: float = 120.0
  """Maximum τ_max in Nm (physical upper bound)."""

  codesign_lr: float = 3e-3
  """Learning rate for the codesign optimizer."""

  codesign_interval: int = 5
  """Update codesign params every N PPO iterations."""

  freeze_tau: bool = False
  """If True, fix τ_max values (use init_tau_max as a fixed catalog) and only
  learn the assignment (alpha). This prevents the optimizer from collapsing
  all types to the same value."""


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
    if cfg.freeze_tau:
      # Fixed motor catalog — only learn assignment, not motor capacities.
      self.register_buffer("tau_raw", init_raw)
    else:
      self.tau_raw = nn.Parameter(init_raw)

    # Temperature state (not a parameter — managed externally).
    self.register_buffer(
      "temperature", torch.tensor(cfg.temperature_init, dtype=torch.float)
    )

    # Torque log buffer for codesign step (detached rollout data).
    self._torque_log: list[torch.Tensor] = []

    # Monitoring metrics from the latest surrogate-loss evaluation.
    self.latest_metrics: dict[str, float] = {}

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
    """Differentiable demand-driven clustering loss for codesign.

    Treats motor sizing as a covering/clustering problem:
      - Each joint has a torque *demand* (high quantile of |unclipped torque|).
      - Each motor type k has a capacity τ_max_k (learnable, unless frozen).
      - Assignment p_uk clusters joints into types.

    Loss terms:
      1. Deficit (covering): penalizes τ_eff < demand → drives τ_max UP to cover
         the demand of assigned joints. This is the upward pressure that the old
         surrogate lacked (which caused all types to collapse to the minimum).
      2. Motor cost: penalizes large active motors → drives τ_max DOWN and merges
         clusters. Balanced against the deficit term.
      3. Type count: penalizes the number of active motor types.

    Returns:
      Scalar loss for backprop through alpha (and tau_raw if not frozen).
    """
    if not self._torque_log:
      return torch.tensor(0.0, device=device, requires_grad=True)

    # Stack logged (unclipped) torques: (N_total, n_joints).
    tau_logged = torch.cat(self._torque_log, dim=0).to(device).abs()

    tau_ref = self.cfg.max_tau

    # --- Per-joint demand: high quantile + safety margin, capped at max_tau. ---
    # Detached: demand is a target the design must cover, not something to game.
    with torch.no_grad():
      q = self.cfg.demand_quantile
      demand_full = torch.quantile(tau_logged, q, dim=0)  # (n_joints,)
      demand_full = (demand_full * self.cfg.demand_margin).clamp(max=tau_ref)
      demand_unique = self.symmetry.reduce_to_unique_max(demand_full)  # (n_unique,)

    # --- Differentiable effective capacity per unique joint. ---
    # Use straight-through hard Gumbel during training so the deficit/coverage is
    # evaluated on the DISCRETE motor that will actually be selected (argmax),
    # not a soft blend. A soft blend could "cover" demand via a mixture while the
    # eventual hard assignment picks a smaller motor and violates coverage.
    temp = float(self.temperature.item())
    if self.training:
      p = F.gumbel_softmax(self.alpha, tau=temp, hard=True)
    else:
      p = F.softmax(self.alpha / max(temp, 0.01), dim=-1)
    tau_eff_unique = (p * self.tau_max.unsqueeze(0)).sum(dim=-1)  # (n_unique,)

    # --- Deficit (covering): upward pressure on τ_max. ---
    # Each unique joint must be covered: τ_eff_j ≥ demand_j. Linear (L1) so the
    # upward gradient is constant whenever undersized — quadratic deficit would
    # vanish near the boundary and let the cost term pull τ_eff below demand.
    deficit = F.relu(demand_unique - tau_eff_unique)
    l_deficit = self.cfg.lambda_deficit * (deficit / tau_ref).mean()

    # --- Assignment cost: per-joint downward pressure on capacity. ---
    # Each joint "pays" for the motor capacity it is assigned, so low-demand
    # joints (e.g. ankle_roll) prefer small motors instead of being assigned to
    # an oversized one. This prevents collapse to a single big motor. With a
    # linear deficit dominating it, the equilibrium sits at τ_eff ≈ demand.
    l_assign_cost = self.cfg.lambda_motor_cost * (tau_eff_unique / tau_ref).mean()

    # --- Type count: fewer distinct motor types (favor reuse/clustering). ---
    usage = p.mean(dim=0)  # (K,)
    active = torch.sigmoid(
      self.cfg.usage_sharpness * (usage - self.cfg.usage_threshold)
    )
    l_types = self.cfg.lambda_types * active.sum()

    # Stash monitoring metrics (detached).
    with torch.no_grad():
      self.latest_metrics = {
        "deficit_mean": deficit.mean().item(),
        "deficit_max": deficit.max().item(),
        "demand_max": demand_unique.max().item(),
        "demand_mean": demand_unique.mean().item(),
        "n_active_types": float((active > 0.5).sum().item()),
      }

    return l_deficit + l_assign_cost + l_types

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
