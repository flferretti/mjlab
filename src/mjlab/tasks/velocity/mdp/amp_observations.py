"""AMP observation functions for the velocity task.

These provide ground-truth (no noise, no corruption) joint and base state
observations consumed by the AMP discriminator.  The ordering must match the
expert motion data produced by ``AMPLoader``:

    joint_pos (N) | joint_vel (N) | base_lin_vel (3) | base_ang_vel (3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_pos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Raw joint positions (absolute, not relative to default)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Raw joint velocities (absolute, not relative to default)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]
