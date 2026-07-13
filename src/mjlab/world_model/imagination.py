"""Imagination-based design evaluation with a trained world model.

:class:`DesignEvaluator` scores actuator designs entirely inside the latent
world model: it encodes design-independent post-reset observations, overwrites
the design (``motor_tau_max``) and command slots of the observation, and rolls
the *actual frozen policy* through the learned dynamics — the same
(policy o dynamics)(theta) system the GA's true-sim backend evaluates, at a
tiny fraction of the cost. A whole population x seeds x ensemble batch is
evaluated in a few forward passes.

Outputs per design: mean step reward (the GA's ``RewardMetric``), per-joint
RMS applied torque (the GA's thermal constraint input), and the ensemble
standard deviation (epistemic uncertainty, used to trigger true-sim
verification).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.world_model.config import ImaginationCfg
from mjlab.world_model.design_space import DesignSpaceCfg, design_space_from_dict
from mjlab.world_model.networks import WorldModel
from mjlab.world_model.replay import Normalizer
from mjlab.world_model.trainer import load_world_model


@dataclass
class DesignScores:
  """Per-design imagination results (all on CPU)."""

  performance: torch.Tensor
  """(L,) ensemble-reduced mean step reward."""
  rms_torque: torch.Tensor
  """(L, J) per-joint RMS applied torque in Nm."""
  uncertainty: torch.Tensor
  """(L,) ensemble std of the performance estimate."""


class DesignEvaluator:
  def __init__(
    self,
    checkpoint: str | Path,
    policy: torch.nn.Module,
    device: torch.device | str,
    cfg: ImaginationCfg | None = None,
    seed: int = 0,
  ) -> None:
    self.cfg = cfg or ImaginationCfg()
    self.device = torch.device(device)
    self.model, self.normalizer, ckpt = load_world_model(checkpoint, self.device)
    self.policy = policy.to(self.device).eval()

    meta = ckpt["dataset_meta"]
    self.meta = meta
    self.design_space: DesignSpaceCfg = design_space_from_dict(meta["design_space"])
    start_obs = ckpt.get("start_obs")
    if start_obs is None:
      raise ValueError("Checkpoint carries no start-observation bank.")
    self.start_obs = start_obs.to(self.device, torch.float32)

    layout = meta["obs_layout"]
    if "motor_tau_max" not in layout:
      raise ValueError(
        "The dataset's actor observation has no 'motor_tau_max' term; a "
        "design-conditioned (MotorCond) task is required."
      )
    self._tau_slice = slice(*layout["motor_tau_max"])
    self._tau_obs_joints: list[str] = meta["tau_obs_joint_names"]
    lo, hi = meta["tau_obs_range"]
    self._tau_obs_range: tuple[float, float] = (float(lo), float(hi))
    self._cmd_slice = slice(*layout["command"]) if "command" in layout else None
    self._gen = torch.Generator(device="cpu").manual_seed(seed)

  # -- Public API ---------------------------------------------------------------

  @torch.no_grad()
  def evaluate_designs(self, theta: torch.Tensor, n_seeds: int) -> DesignScores:
    """Score ``(L, D)`` designs from ``n_seeds`` imagined start states each."""
    cfg = self.cfg
    model, norm = self.model, self.normalizer
    L = theta.shape[0]
    B = L * n_seeds
    theta = theta.to(self.device, torch.float32)
    theta_rep = theta.repeat_interleave(n_seeds, dim=0)  # [B, D]
    e = model.embed_design(self.design_space.normalize(theta_rep))
    tau_obs = self._tau_obs_block(theta_rep)  # [B, n_joints_obs]
    command = torch.tensor(cfg.command, device=self.device).expand(B, -1)

    K = model.cfg.ensemble_size
    reward_sum = torch.zeros(K, B, device=self.device)
    torque_sq_sum = torch.zeros(K, B, model.dims.num_joints, device=self.device)
    obs_raw = self._sample_starts(B)
    obs_raw = self._overwrite_slots(obs_raw, tau_obs, command)
    z = [self._encode_raw(obs_raw, e) for _ in range(K)]

    tau_std = norm.stats["torque_std"]
    r_std = norm.stats["reward_std"]
    cmd_in = command if self._uses_command() else None

    for _ in range(cfg.rollout_steps):
      for k in range(K):
        obs_pred = norm.denormalize_obs(model.predict_obs(z[k], e))
        obs_pred = self._overwrite_slots(obs_pred, tau_obs, command)
        action = self.policy(obs_pred)
        reward_sum[k] += model.reward(k, z[k], action, cmd_in, e)[..., 0] * r_std
        torque_sq_sum[k] += (model.torque(z[k], action, e) * tau_std) ** 2
        z_next, _ = model.step(k, z[k], action, cmd_in, e)
        if cfg.reset_on_termination:
          p_term = torch.sigmoid(model.termination(z_next, e))
          reset = p_term > cfg.termination_threshold
          if bool(reset.any()):
            fresh = self._sample_starts(int(reset.sum()))
            fresh = self._overwrite_slots(fresh, tau_obs[reset], command[reset])
            z_next[reset] = self._encode_raw(fresh, e[reset])
        z[k] = z_next

    steps = cfg.rollout_steps
    per_member = (reward_sum / steps).view(K, L, n_seeds).mean(dim=-1)  # [K, L]
    rms = (
      (torque_sq_sum / steps).sqrt().view(K, L, n_seeds, -1).mean(dim=2)  # [K, L, J]
    )
    if cfg.ensemble_reduce == "median":
      perf = per_member.median(dim=0).values
      rms_red = rms.median(dim=0).values
    else:
      perf = per_member.mean(dim=0)
      rms_red = rms.mean(dim=0)
    return DesignScores(
      performance=perf.cpu(),
      rms_torque=rms_red.cpu(),
      uncertainty=per_member.std(dim=0).cpu(),
    )

  @torch.no_grad()
  def evaluate_joint_tau(
    self,
    tau_LJ: torch.Tensor,
    joint_order: list[str] | tuple[str, ...],
    n_seeds: int,
  ) -> DesignScores:
    """Score per-joint torque designs in the GA's ``joint_order`` layout."""
    theta = self.design_space.theta_from_joint_tau(
      tau_LJ.to(self.device, torch.float32), joint_order
    )
    return self.evaluate_designs(theta, n_seeds)

  # -- Internals ------------------------------------------------------------------

  def _uses_command(self) -> bool:
    return self.model.dims.command_dim > 0

  def _sample_starts(self, n: int) -> torch.Tensor:
    idx = torch.randint(self.start_obs.shape[0], (n,), generator=self._gen)
    return self.start_obs[idx.to(self.device)].clone()

  def _encode_raw(self, obs_raw: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    obs_n = self.normalizer.normalize_obs(obs_raw)
    if self.model.cfg.privileged_encoder:
      # The privileged-encoder ablation self-conditions on the contact head's
      # prediction; at the start state we have no latent yet, so a zero
      # contact-feature vector (mid-flight prior) seeds the first encoding.
      from mjlab.world_model.networks import contact_feature_dim

      contact = obs_raw.new_zeros(
        obs_raw.shape[0], contact_feature_dim(self.model.dims.num_contacts)
      )
      return self.model.encode(obs_n, contact)
    return self.model.encode(obs_n)

  def _tau_obs_block(self, theta_rep: torch.Tensor) -> torch.Tensor:
    """Normalized per-joint tau observation, in the policy's joint order."""
    tau_joints = self.design_space.joint_tau_from_theta(theta_rep)
    space_order = list(self.design_space.joint_names)
    idx = [space_order.index(j) for j in self._tau_obs_joints]
    tau = tau_joints[:, idx]
    lo, hi = self._tau_obs_range
    return ((tau - lo) / (hi - lo)).clamp(0.0, 1.0)

  def _overwrite_slots(
    self, obs_raw: torch.Tensor, tau_obs: torch.Tensor, command: torch.Tensor
  ) -> torch.Tensor:
    obs_raw[:, self._tau_slice] = tau_obs
    if self._cmd_slice is not None:
      obs_raw[:, self._cmd_slice] = command
    return obs_raw


__all__ = ["DesignEvaluator", "DesignScores", "Normalizer", "WorldModel"]
