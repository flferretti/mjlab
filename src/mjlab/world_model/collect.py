"""Design-randomized trajectory collection for world-model training.

Builds a motor-conditioned task env in which every environment carries a
different actuator design (resampled per reset via
:func:`mjlab.world_model.design_space.randomize_design`), rolls a frozen
design-conditioned policy with an exploration mixture, and writes trajectory
shards in the :mod:`mjlab.world_model.replay` layout.

Physics domain randomization is disabled (matching the eval configuration of
``scripts/codesign_ga.py``'s ``MjlabBackend``) so the world model learns the
same (policy o dynamics)(theta) system the GA queries; command randomization
stays on so the model covers the command space.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg
from mjlab.world_model.config import CollectCfg
from mjlab.world_model.design_space import (
  DesignSpaceCfg,
  current_design,
  randomize_design,
)
from mjlab.world_model.replay import ShardWriter

# Randomization events popped from the task config, mirroring MjlabBackend in
# scripts/codesign_ga.py (the GA's eval env), so the surrogate target is clean.
_DISABLED_EVENTS = (
  "encoder_bias",
  "base_com",
  "foot_friction_slide",
  "randomize_robot_mass",
  "randomize_actuator_gains",
  "randomize_joint_friction",
  "joint_default_pos_noise",
  "randomize_terrain",
  "randomize_motor_tau_max",
  "push_robot",
)

_START_OBS_BANK_SIZE = 4096


class RandomPolicy(torch.nn.Module):
  """Uniform random actions in [-1, 1] (smoke tests, exploration bucket)."""

  def __init__(self, action_dim: int) -> None:
    super().__init__()
    self.action_dim = action_dim

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.empty(obs.shape[0], self.action_dim, device=obs.device).uniform_(
      -1.0, 1.0
    )


class FlatActorPolicy(torch.nn.Module):
  """Wrap an rsl_rl actor so it accepts a flat observation tensor."""

  def __init__(self, actor: torch.nn.Module) -> None:
    super().__init__()
    self.obs_normalizer = cast(torch.nn.Module, actor.obs_normalizer)
    self.mlp = cast(torch.nn.Module, actor.mlp)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(self.obs_normalizer(obs))


def load_policy(
  task: str, policy_path: str | None, device: str, action_dim: int
) -> torch.nn.Module:
  """Load a frozen policy: TorchScript, rsl_rl checkpoint, or random.

  Mirrors the loader in ``scripts/codesign_ga.py`` so the collector rolls the
  exact policy the GA evaluates with.
  """
  if policy_path is None:
    return RandomPolicy(action_dim).to(device)
  try:
    return torch.jit.load(policy_path, map_location=device).eval()
  except (RuntimeError, ValueError) as exc:
    msg = str(exc)
    if "constants.pkl" not in msg and "PytorchStreamReader" not in msg:
      raise

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.amp_runner import MjlabAmpOnPolicyRunner
  from mjlab.tasks.registry import load_rl_cfg

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


class BehaviorMixture(torch.nn.Module):
  """Frozen policy plus per-env exploration noise buckets.

  Env buckets (fixed for the whole collection):
    - ``policy_fraction``: policy actions plus i.i.d. Gaussian noise with a
      std cycled from ``policy_noise_stds``.
    - ``correlated_noise_fraction``: policy actions plus AR(1) noise, which
      drags the state slightly off the policy manifold without leaving the
      evaluation distribution.
    - remainder: uniform random actions.
  """

  def __init__(
    self,
    policy: torch.nn.Module,
    num_envs: int,
    action_dim: int,
    cfg: CollectCfg,
    device: str,
  ) -> None:
    super().__init__()
    self.policy = policy
    self.cfg = cfg
    n_policy = int(round(cfg.policy_fraction * num_envs))
    n_corr = int(round(cfg.correlated_noise_fraction * num_envs))
    n_policy = min(n_policy, num_envs)
    n_corr = min(n_corr, num_envs - n_policy)
    self.n_policy, self.n_corr = n_policy, n_corr
    stds = torch.as_tensor(cfg.policy_noise_stds, device=device)
    self.policy_noise_std = stds[torch.arange(n_policy, device=device) % len(stds)]
    self.corr_state = torch.zeros(n_corr, action_dim, device=device)

  @torch.no_grad()
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    actions = self.policy(obs)
    n_p, n_c = self.n_policy, self.n_corr
    if n_p > 0:
      noise = torch.randn_like(actions[:n_p]) * self.policy_noise_std[:, None]
      actions[:n_p] = actions[:n_p] + noise
    if n_c > 0:
      beta = self.cfg.correlated_noise_beta
      self.corr_state.mul_(beta).add_(
        torch.randn_like(self.corr_state),
        alpha=self.cfg.correlated_noise_std * (1.0 - beta**2) ** 0.5,
      )
      actions[n_p : n_p + n_c] = actions[n_p : n_p + n_c] + self.corr_state
    if n_p + n_c < actions.shape[0]:
      actions[n_p + n_c :] = torch.empty_like(actions[n_p + n_c :]).uniform_(-1, 1)
    return actions


def build_collect_env(
  task: str,
  cfg: CollectCfg,
  design_space: DesignSpaceCfg,
  device: str,
  with_design_event: bool = True,
) -> ManagerBasedRlEnv:
  """Task env with physics DR off and per-reset design randomization on.

  With ``with_design_event=False`` no design event is installed (designs are
  then injected explicitly via ``design_space.apply``, e.g. for ground-truth
  rollouts of fixed designs).
  """
  env_cfg = load_env_cfg(task, play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if getattr(env_cfg, "events", None) is not None:
    for name in _DISABLED_EVENTS:
      env_cfg.events.pop(name, None)
  if with_design_event:
    # The holdout carve-out is seeded independently of the collection seed so
    # train/interp/extrap splits agree across collection runs.
    env_cfg.events["randomize_design"] = EventTermCfg(
      mode="reset",
      func=randomize_design,
      params={
        "design_space": design_space,
        "region": cfg.split,
        "n_holdout": cfg.n_holdout,
        "holdout_seed": 0,
      },
    )
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _flat_actor_obs(obs: dict) -> torch.Tensor:
  actor = obs["actor"]
  assert isinstance(actor, torch.Tensor), "actor obs group must be flat"
  return actor


def _command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


def _contact_fields(
  sensor: ContactSensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  data = sensor.data
  assert data.found is not None and data.force is not None, (
    "The contact sensor must be configured with fields=('found', 'force')."
  )
  assert data.current_air_time is not None, (
    "The contact sensor must be configured with track_air_time=True."
  )
  assert data.current_contact_time is not None
  return data.found, data.force, data.current_air_time, data.current_contact_time


def _tau_obs_meta(env: ManagerBasedRlEnv) -> dict[str, Any]:
  """Joint order and range of the policy's ``motor_tau_max`` observation."""
  actor_group = env.cfg.observations.get("actor")
  term = getattr(actor_group, "terms", {}).get("motor_tau_max")
  if term is None:
    return {"tau_obs_joint_names": None, "tau_obs_range": None}
  return {
    "tau_obs_joint_names": list(term.params["joint_names"]),
    "tau_obs_range": list(term.params["tau_range"]),
  }


def _actor_obs_layout(env: ManagerBasedRlEnv) -> dict[str, list[int]]:
  """Term name -> [start, end) slice into the flat actor observation."""
  names = env.observation_manager.active_terms["actor"]
  dims = env.observation_manager.group_obs_term_dim["actor"]
  layout: dict[str, list[int]] = {}
  offset = 0
  for name, dim in zip(names, dims, strict=True):
    size = int(np.prod(dim))
    layout[name] = [offset, offset + size]
    offset += size
  return layout


def collect_dataset(
  task: str,
  cfg: CollectCfg,
  design_space: DesignSpaceCfg,
  out_dir: str | Path,
  policy_path: str | None,
  device: str,
  contact_sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
) -> dict[str, Any]:
  """Roll the behavior mixture and write a trajectory dataset.

  Returns the dataset meta dict (also written to ``meta.json``).
  """
  torch.manual_seed(cfg.seed)
  env = build_collect_env(task, cfg, design_space, device)
  try:
    return _collect_into(
      env,
      task,
      cfg,
      design_space,
      out_dir,
      policy_path,
      device,
      contact_sensor_name,
      command_name,
    )
  finally:
    env.close()


def _collect_into(
  env: ManagerBasedRlEnv,
  task: str,
  cfg: CollectCfg,
  design_space: DesignSpaceCfg,
  out_dir: str | Path,
  policy_path: str | None,
  device: str,
  contact_sensor_name: str,
  command_name: str,
) -> dict[str, Any]:
  action_dim = env.action_manager.total_action_dim
  policy = load_policy(task, policy_path, device, action_dim)
  behavior = BehaviorMixture(policy, cfg.num_envs, action_dim, cfg, device)

  robot = env.scene["robot"]
  torque_joint_ids, _ = robot.find_joints(
    list(design_space.joint_names), preserve_order=True
  )
  sensor = env.scene[contact_sensor_name]
  assert isinstance(sensor, ContactSensor), (
    f"'{contact_sensor_name}' is not a ContactSensor."
  )
  reward_term_names = list(env.reward_manager._term_names)

  obs, _ = env.reset()
  actor_obs = _flat_actor_obs(obs)
  num_contacts = _contact_fields(sensor)[0].shape[1]
  has_command = command_name in env.command_manager.active_terms
  command_dim = _command(env, command_name).shape[1] if has_command else 0

  field_shapes: dict[str, tuple[int, ...]] = {
    "obs": (actor_obs.shape[1],),
    "action": (action_dim,),
    "reward": (),
    "reward_terms": (len(reward_term_names),),
    "terminated": (),
    "done": (),
    "contact_found": (num_contacts,),
    "contact_force": (num_contacts, 3),
    "air_time": (num_contacts,),
    "contact_time": (num_contacts,),
    "torque": (len(torque_joint_ids),),
    "theta": (design_space.dim,),
    "command": (command_dim,),
  }
  store_dtype = torch.float16 if cfg.store_dtype == "float16" else torch.float32
  writer = ShardWriter(
    out_dir,
    cfg.num_envs,
    field_shapes,
    device=device,
    flush_steps=min(cfg.flush_steps, cfg.steps),
    shard_steps=cfg.shard_steps,
    store_dtype=store_dtype,
  )

  start_obs_bank = [actor_obs.detach().cpu()]
  bank_rows = actor_obs.shape[0]
  prev_done = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

  for _ in range(cfg.steps):
    # State-aligned fields (pre-step, matching the obs the action sees).
    found, force, air_time, contact_time = _contact_fields(sensor)
    row: dict[str, torch.Tensor] = {
      "obs": actor_obs,
      "contact_found": (found > 0).float(),
      "contact_force": force,
      "air_time": air_time,
      "contact_time": contact_time,
      "theta": current_design(env, design_space),
      "command": (
        _command(env, command_name)
        if has_command
        else actor_obs.new_zeros(cfg.num_envs, 0)
      ),
    }
    actions = behavior(actor_obs)
    obs, reward, terminated, truncated, _ = env.step(actions)
    actor_obs = _flat_actor_obs(obs)
    done = terminated | truncated
    # Transition fields (the step just executed).
    row.update(
      action=actions,
      reward=reward,
      reward_terms=env.reward_manager._step_reward,
      torque=robot.data.qfrc_actuator[:, torque_joint_ids],
      terminated=terminated.float(),
      done=done.float(),
    )
    writer.add(row)

    if bool(prev_done.any()) and bank_rows < _START_OBS_BANK_SIZE:
      fresh = actor_obs[prev_done].detach().cpu()
      start_obs_bank.append(fresh)
      bank_rows += fresh.shape[0]
    prev_done = done

  writer.save_extra(
    "start_obs.pt", torch.cat(start_obs_bank, dim=0)[:_START_OBS_BANK_SIZE]
  )
  meta: dict[str, Any] = {
    "task": task,
    "backend": "mujoco_warp",
    "mjlab_version": importlib.metadata.version("mjlab"),
    "policy_path": policy_path,
    "collect_cfg": asdict(cfg),
    "design_space": asdict(design_space),
    "design_space_name": design_space.name,
    "reward_term_names": reward_term_names,
    "obs_layout": _actor_obs_layout(env),
    **_tau_obs_meta(env),
    "contact_sensor": contact_sensor_name,
    "command_name": command_name if has_command else None,
    "torque_joint_names": list(design_space.joint_names),
    "step_dt": env.step_dt,
  }
  writer.close(meta)
  return meta
