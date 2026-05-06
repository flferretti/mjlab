"""Reset event term that initializes robot state from AMP motion data.

This samples random frames from the Mixamo expert motion clips so that the
robot starts each episode in a realistic walking pose, matching the
``reset_state_from_amp`` event in ``gb-rl-locomotion``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from amp_rsl_rl.utils import AMPLoader

from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class reset_state_from_amp:
  """Reset the robot state from AMP motion data on episode reset."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    dataset_cfg: dict = cfg.params["dataset_cfg"]

    sim_dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation
    self.amp_data = AMPLoader(
      device=env.device,
      dataset_path_root=dataset_cfg["amp_data_path"],
      datasets=dataset_cfg["datasets"],
      simulation_dt=sim_dt,
      slow_down_factor=dataset_cfg["slow_down_factor"],
      expected_joint_names=dataset_cfg["amp_joint_names"],
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    pass

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    base_lin_vel_range: tuple[float, float],
    base_ang_vel_range: tuple[float, float],
    dataset_cfg: dict,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]

    (
      orientations,
      amp_joint_pos,
      amp_joint_vel,
      base_lin_vel,
      base_ang_vel,
    ) = self.amp_data.get_state_for_reset(num_envs)

    amp_joint_pos += sample_uniform(*position_range, amp_joint_pos.shape, env.device)
    amp_joint_vel += sample_uniform(*velocity_range, amp_joint_vel.shape, env.device)

    amp_joint_names: list[str] = dataset_cfg["amp_joint_names"]
    entity_joint_names = asset.joint_names
    amp_joint_indices = [entity_joint_names.index(name) for name in amp_joint_names]

    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    joint_pos = default_joint_pos[env_ids].clone()
    joint_vel = sample_uniform(
      *velocity_range, (num_envs, len(entity_joint_names)), env.device
    )

    joint_pos[:, amp_joint_indices] = amp_joint_pos
    joint_vel[:, amp_joint_indices] = amp_joint_vel

    pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(pos_limits[..., 0], pos_limits[..., 1])

    # Clamp velocities to reasonable bounds.
    max_vel = 20.0
    joint_vel = joint_vel.clamp_(-max_vel, max_vel)

    base_lin_vel += sample_uniform(*base_lin_vel_range, base_lin_vel.shape, env.device)
    base_ang_vel += sample_uniform(*base_ang_vel_range, base_ang_vel.shape, env.device)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.write_root_link_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
      torch.cat([base_lin_vel, base_ang_vel], dim=-1), env_ids=env_ids
    )
