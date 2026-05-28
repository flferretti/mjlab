"""PD actuator with differentiable motor-type codesign.

Replaces hard torque clipping with soft saturation from a
GumbelSoftmaxActuator. During rollout, torques are logged for the
codesign optimizer step. The codesign scheduler (CodesignScheduler)
runs as an alternating optimization alongside PPO.

Usage:
  1. Configure CodesignPdActuatorCfg with motor types and symmetry
  2. Training loop calls actuator.compute() normally
  3. After PPO update, call scheduler.step(ppo_iter)
  4. At convergence, extract hard assignment with gumbel.hard_assignment()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class CodesignPdActuatorCfg(IdealPdActuatorCfg):
  """Configuration for PD actuator with motor-type codesign.

  Extends IdealPdActuatorCfg with codesign parameters. The effort_limit
  from the base class is used as the initial upper bound (max_tau).
  """

  n_types: int = 2
  """Number of motor types to learn."""

  init_tau_max: list[float] = field(default_factory=lambda: [80.0, 40.0])
  """Initial τ_max per type (Nm). Length must equal n_types."""

  joint_names: list[str] = field(default_factory=list)
  """Joint names in order (required for symmetry expansion)."""

  symmetry_pairs: list[tuple[str, str]] = field(default_factory=list)
  """(left, right) joint pairs that share motor assignment."""

  temperature_init: float = 1.0
  """Initial Gumbel-Softmax temperature."""

  temperature_min: float = 0.1
  """Minimum temperature."""

  temperature_decay: float = 0.9995
  """Multiplicative temperature decay per iteration."""

  lambda_types: float = 0.01
  """Weight: fewer motor types."""

  lambda_balance: float = 0.05
  """Weight: prevent outlier assignments."""

  lambda_tau: float = 1e-4
  """Weight: downward pressure on τ_max."""

  lambda_saturation: float = 0.1
  """Weight: penalize operating near saturation."""

  lambda_rms: float = 0.05
  """Weight: minimize RMS torque in surrogate."""

  lambda_peak: float = 0.1
  """Weight: minimize peak torque in surrogate."""

  min_tau: float = 5.0
  """Minimum τ_max (Nm)."""

  max_tau: float = 120.0
  """Maximum τ_max (Nm)."""

  codesign_lr: float = 3e-3
  """Learning rate for codesign optimizer."""

  codesign_interval: int = 5
  """Update codesign params every N PPO iterations."""

  def build(
    self, entity: "Entity", target_ids: list[int], target_names: list[str]
  ) -> "CodesignPdActuator":
    return CodesignPdActuator(self, entity, target_ids, target_names)


class CodesignPdActuator(IdealPdActuator["CodesignPdActuatorCfg"]):
  """PD actuator with differentiable motor-type codesign.

  Overrides _clip_effort to use soft saturation via GumbelSoftmaxActuator.
  Logs torques during rollout for the alternating codesign optimization.
  """

  cfg: CodesignPdActuatorCfg

  def __init__(
    self,
    cfg: CodesignPdActuatorCfg,
    entity: "Entity",
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)

    from mjlab.actuator.gumbel_codesign import (
      CodesignConfig,
      CodesignScheduler,
      GumbelSoftmaxActuator,
      SymmetrySpec,
    )

    symmetry = SymmetrySpec(
      joint_names=cfg.joint_names or target_names,
      symmetry_pairs=cfg.symmetry_pairs,
    )

    codesign_cfg = CodesignConfig(
      n_types=cfg.n_types,
      init_tau_max=cfg.init_tau_max,
      temperature_init=cfg.temperature_init,
      temperature_min=cfg.temperature_min,
      temperature_decay=cfg.temperature_decay,
      lambda_types=cfg.lambda_types,
      lambda_balance=cfg.lambda_balance,
      lambda_tau=cfg.lambda_tau,
      lambda_saturation=cfg.lambda_saturation,
      lambda_rms=cfg.lambda_rms,
      lambda_peak=cfg.lambda_peak,
      min_tau=cfg.min_tau,
      max_tau=cfg.max_tau,
      codesign_lr=cfg.codesign_lr,
      codesign_interval=cfg.codesign_interval,
    )

    self.gumbel = GumbelSoftmaxActuator(codesign_cfg, symmetry)
    self.gumbel.eval()
    self._scheduler: CodesignScheduler | None = None

  def initialize(self, mj_model, model, data, device) -> None:
    super().initialize(mj_model, model, data, device)
    self.gumbel = self.gumbel.to(device)

    from mjlab.actuator.gumbel_codesign import CodesignScheduler

    self._scheduler = CodesignScheduler(self.gumbel, device=device)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    """Replace hard clamp with soft saturation and log torques."""
    self.gumbel.log_torques(effort)
    return self.gumbel(effort)

  def codesign_step(self, ppo_iter: int) -> dict[str, float] | None:
    """Run one codesign optimizer step. Call after PPO update.

    Args:
      ppo_iter: Current PPO iteration number.

    Returns:
      Dict of loss components if update was performed, None otherwise.
    """
    if self._scheduler is None:
      return None
    return self._scheduler.step(ppo_iter)

  @property
  def codesign_summary(self) -> str:
    """Human-readable summary of motor assignment state."""
    return self.gumbel.summary()
