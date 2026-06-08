"""Per-joint max-torque randomization for motor-conditioned policies.

This module supports training a policy that is *conditioned* on the maximum
torque available at each joint. At every reset, each environment samples a new
per-joint ``tau_max`` from a fixed range; these limits are written into the
actuators (so the physics actually clips torque at the sampled value) and are
exposed to the policy through an observation term. A policy trained this way can
run with arbitrary motor configurations, which a downstream genetic algorithm
can then exploit to search for the best torque combination without retraining.

The randomization can enforce left/right symmetry: a single value is sampled per
symmetric joint group and mirrored onto both sides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_BUFFER_ATTR = "_motor_tau_max"
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _group_index(
  joint_names: list[str],
  symmetry_pairs: list[tuple[str, str]],
  device: str | torch.device,
) -> tuple[torch.Tensor, int]:
  """Map each joint to a sampling group, collapsing symmetric pairs.

  Returns a ``(n_joints,)`` long tensor of group ids and the number of groups.
  With no symmetry pairs every joint is its own group.
  """
  right_to_left = {right: left for left, right in symmetry_pairs}
  unique: list[str] = []
  group_of_joint: list[int] = []
  for name in joint_names:
    canon = right_to_left.get(name, name)
    if canon not in unique:
      unique.append(canon)
    group_of_joint.append(unique.index(canon))
  return torch.tensor(group_of_joint, device=device, dtype=torch.long), len(unique)


def _get_or_init_buffer(
  env: "ManagerBasedRlEnv",
  joint_names: list[str],
  tau_range: tuple[float, float],
) -> torch.Tensor:
  """Return the per-env, per-joint tau_max buffer, creating it if needed.

  Initialized to the midpoint of ``tau_range`` so the observation has a valid
  value even before the first reset event runs (e.g. during the observation
  manager's dimension probe).
  """
  buffer = getattr(env, _BUFFER_ATTR, None)
  if buffer is None or buffer.shape != (env.num_envs, len(joint_names)):
    midpoint = 0.5 * (tau_range[0] + tau_range[1])
    buffer = torch.full(
      (env.num_envs, len(joint_names)),
      midpoint,
      device=env.device,
      dtype=torch.float,
    )
    setattr(env, _BUFFER_ATTR, buffer)
  return buffer


def set_motor_tau_max(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  tau_values: torch.Tensor,
  joint_names: list[str],
  tau_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply explicit per-joint max torques to the given environments.

  Single source of truth for writing ``tau_max``: updates the observation
  buffer AND the actuators' effort limits so the simulation clips torque at the
  requested values. Used by the randomization event during training and by the
  genetic-algorithm evaluation (Phase 2) to inject fixed genomes.

  Args:
    env_ids: Environments to update.
    tau_values: ``(len(env_ids), n_joints)`` torque limits in Nm, in
      ``joint_names`` order.
    joint_names: Canonical joint order defining the buffer/observation layout.
    tau_range: ``(min, max)`` range (used to size/initialize the buffer).
    asset_cfg: Entity whose actuators receive the limits.
  """
  buffer = _get_or_init_buffer(env, joint_names, tau_range)
  buffer[env_ids] = tau_values.to(buffer.dtype)

  name_to_idx = {name: i for i, name in enumerate(joint_names)}
  asset: "Entity" = env.scene[asset_cfg.name]
  for actuator in asset.actuators:
    force_limit = getattr(actuator, "force_limit", None)
    if force_limit is None:
      continue
    for i, jname in enumerate(actuator.target_names):
      j = name_to_idx.get(jname)
      if j is not None:
        force_limit[env_ids, i] = buffer[env_ids, j]


def randomize_motor_tau_max(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  joint_names: list[str],
  tau_range: tuple[float, float],
  symmetry_pairs: list[tuple[str, str]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Sample new per-joint max torques and apply them to the actuators.

  Intended as a ``reset``-mode event. For the resetting environments it samples
  a ``tau_max`` per joint (optionally shared across symmetric pairs) and applies
  it via :func:`set_motor_tau_max`, so both the observation and the physical
  effort limits reflect the new motor configuration.

  Args:
    env_ids: Environments to resample (``None`` means all).
    joint_names: Canonical joint order defining the buffer/observation layout.
    tau_range: ``(min, max)`` torque limit in Nm to sample uniformly from.
    symmetry_pairs: ``(left, right)`` joint pairs that share a sampled value.
    asset_cfg: Entity whose actuators receive the limits.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  group_idx, n_groups = _group_index(joint_names, symmetry_pairs or [], env.device)
  samples = torch.empty((env_ids.shape[0], n_groups), device=env.device).uniform_(
    tau_range[0], tau_range[1]
  )
  tau_joints = samples[:, group_idx]  # (n_reset, n_joints)
  set_motor_tau_max(env, env_ids, tau_joints, joint_names, tau_range, asset_cfg)


def motor_tau_max(
  env: "ManagerBasedRlEnv",
  joint_names: list[str],
  tau_range: tuple[float, float],
) -> torch.Tensor:
  """Observation: per-joint max torque normalized to ``[0, 1]``.

  Returns a ``(num_envs, n_joints)`` tensor giving the currently configured
  ``tau_max`` for each joint, normalized by ``tau_range`` so the policy receives
  a well-scaled motor descriptor.
  """
  buffer = _get_or_init_buffer(env, joint_names, tau_range)
  lo, hi = tau_range
  return (buffer - lo) / (hi - lo)
