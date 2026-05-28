"""Codesign-aware torque reward penalties.

These reward terms penalize torque usage in ways that go beyond simple L2:
- RMS torque: penalizes sustained high effort
- Peak torque: penalizes instantaneous spikes
- Saturation proximity: penalizes operating close to actuator limits
- Torque rate: penalizes sudden changes (jerk avoidance)

All functions follow the mjlab RewardTermCfg protocol:
  func(env, **params) -> torch.Tensor of shape (num_envs,)

These are *policy-facing* penalties: they shape the policy to use less torque.
The motor sizing optimization is handled separately by CodesignScheduler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def torque_rms(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  tau_nominal: float = 80.0,
) -> torch.Tensor:
  """Penalize root-mean-square torque across all joints.

  Unlike L2 (sum of squares), RMS normalizes by number of joints, making
  the penalty scale-invariant to robot size. Normalized by tau_nominal for
  consistent weighting across different robots.

  Args:
    asset_cfg: Robot entity configuration.
    tau_nominal: Reference torque (Nm) for normalization. Use the largest
      motor's rated torque.

  Returns:
    Shape (num_envs,). Always non-negative.
  """
  asset: "Entity" = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.actuator_ids]  # (B, J)
  rms = torch.sqrt(torch.mean(tau**2, dim=1) + 1e-8)  # (B,)
  return rms / tau_nominal


def torque_peak(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  tau_nominal: float = 80.0,
) -> torch.Tensor:
  """Penalize maximum absolute torque across all joints.

  Targets instantaneous peaks that stress individual joints. A robot using
  50Nm on all joints has low RMS but if one joint hits 100Nm that's still
  problematic for motor selection.

  Args:
    asset_cfg: Robot entity configuration.
    tau_nominal: Reference torque (Nm) for normalization.

  Returns:
    Shape (num_envs,). Always non-negative.
  """
  asset: "Entity" = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.actuator_ids]  # (B, J)
  peak = torch.max(torch.abs(tau), dim=1).values  # (B,)
  return peak / tau_nominal


class torque_saturation_proximity:
  """Penalize joints operating close to their effort limits.

  Uses a smooth sigmoid to ramp penalty from 0 (well below limit) to 1
  (at or above limit). The threshold parameter controls where penalty
  starts (default 80% of limit).

  This is critical for codesign: it tells the policy "back off from the
  limits" so the codesign optimizer can reduce motor capacity without
  the policy immediately saturating.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: "Entity" = env.scene[cfg.params["asset_cfg"].name]
    self._actuator_ids = _resolve_actuator_ids(cfg.params["asset_cfg"], asset)
    limits = asset.data.effort_limits  # (num_envs, num_actuators)
    self._limits = limits[:1, self._actuator_ids]  # (1, J)
    self._threshold = float(cfg.params.get("threshold", 0.8))
    self._sharpness = float(cfg.params.get("sharpness", 15.0))

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    threshold: float = 0.8,
    sharpness: float = 15.0,
  ) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, self._actuator_ids]  # (B, J)
    ratio = torch.abs(tau) / (self._limits + 1e-6)
    penalty = torch.sigmoid(self._sharpness * (ratio - self._threshold))
    return penalty.mean(dim=1)  # (B,)


def torque_rate_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  tau_nominal: float = 80.0,
) -> torch.Tensor:
  """Penalize change in torque between timesteps (jerk avoidance).

  Smooth torque profiles allow smaller motors (less peak demand) and
  reduce mechanical wear. Uses the difference between current and
  previous actuator forces.

  Args:
    asset_cfg: Robot entity configuration.
    tau_nominal: Reference torque for normalization.

  Returns:
    Shape (num_envs,).
  """
  asset: "Entity" = env.scene[asset_cfg.name]
  tau_curr = asset.data.actuator_force[:, asset_cfg.actuator_ids]
  if hasattr(asset.data, "prev_actuator_force"):
    tau_prev = asset.data.prev_actuator_force[:, asset_cfg.actuator_ids]
  else:
    tau_prev = tau_curr
  dtau = tau_curr - tau_prev
  return torch.sum(dtau**2, dim=1) / (tau_nominal**2)


class codesign_torque_composite:
  """Combined torque penalty: RMS + peak + saturation + rate.

  All-in-one reward term that avoids needing 4 separate entries in the
  reward config. Each component can be weighted independently.

  Params:
    w_rms: Weight for RMS penalty (default 1.0)
    w_peak: Weight for peak penalty (default 2.0)
    w_saturation: Weight for saturation proximity (default 3.0)
    w_rate: Weight for torque rate (default 0.5)
    tau_nominal: Reference torque for normalization (default 80.0)
    saturation_threshold: Ratio threshold for saturation penalty (default 0.8)
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: "Entity" = env.scene[cfg.params["asset_cfg"].name]
    self._actuator_ids = _resolve_actuator_ids(cfg.params["asset_cfg"], asset)
    self._tau_nominal = float(cfg.params.get("tau_nominal", 80.0))
    self._w_rms = float(cfg.params.get("w_rms", 1.0))
    self._w_peak = float(cfg.params.get("w_peak", 2.0))
    self._w_saturation = float(cfg.params.get("w_saturation", 3.0))
    self._w_rate = float(cfg.params.get("w_rate", 0.5))
    self._sat_threshold = float(cfg.params.get("saturation_threshold", 0.8))
    self._sat_sharpness = float(cfg.params.get("saturation_sharpness", 15.0))

    limits = asset.data.effort_limits
    self._limits = limits[:1, self._actuator_ids]  # (1, J)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
  ) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, self._actuator_ids]  # (B, J)
    B = tau.shape[0]

    result = torch.zeros(B, device=tau.device)

    # RMS component.
    if self._w_rms > 0:
      rms = torch.sqrt(torch.mean(tau**2, dim=1) + 1e-8)
      result = result + self._w_rms * rms / self._tau_nominal

    # Peak component.
    if self._w_peak > 0:
      peak = torch.max(torch.abs(tau), dim=1).values
      result = result + self._w_peak * peak / self._tau_nominal

    # Saturation proximity.
    if self._w_saturation > 0:
      ratio = torch.abs(tau) / (self._limits + 1e-6)
      sat_penalty = torch.sigmoid(self._sat_sharpness * (ratio - self._sat_threshold))
      result = result + self._w_saturation * sat_penalty.mean(dim=1)

    # Torque rate.
    if self._w_rate > 0 and hasattr(asset.data, "prev_actuator_force"):
      tau_prev = asset.data.prev_actuator_force[:, self._actuator_ids]
      dtau = tau - tau_prev
      result = result + self._w_rate * torch.sum(dtau**2, dim=1) / (
        self._tau_nominal**2
      )

    return result


def _resolve_actuator_ids(
  asset_cfg: SceneEntityCfg, asset: "Entity"
) -> list[int] | slice:
  """Resolve actuator IDs from asset config."""
  if asset_cfg.actuator_ids is not None:
    return asset_cfg.actuator_ids
  if asset_cfg.joint_names is not None:
    ids, _ = asset.find_joints(asset_cfg.joint_names)
    return ids
  return slice(None)
