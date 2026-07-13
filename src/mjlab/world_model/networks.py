"""Networks for the contact-enhanced design-conditioned world model.

Architecture summary:
  - Encoder h(o) -> z with SimNorm latents (decoder-free, TD-MPC2-style).
  - Design embedding e(theta) FiLM-conditions the dynamics and every head.
  - Contact-structured dynamics: each ensemble member predicts next-step
    per-contact-point logits from (z, a, cmd); a straight-through
    Gumbel-sigmoid turns them into a discrete mode code whose factored
    embedding FiLM-modulates the dynamics trunk (soft mixture over contact
    modes, supervised by the privileged contact sensors).
  - Shared heads on the latent: current contact state/forces, termination,
    per-joint applied torque (reproduces the GA's RMS thermal constraint),
    and the actor observation (so the actual frozen policy can be rolled out
    inside imagination without a distillation step).

All modules are device-agnostic PyTorch (ROCm-portable).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from mjlab.world_model.config import WorldModelCfg
from mjlab.world_model.simnorm import SimNorm


@dataclass(frozen=True)
class WorldModelDims:
  """Problem dimensions inferred from the dataset."""

  obs_dim: int
  """Flat actor-observation dimension."""
  action_dim: int
  command_dim: int
  num_contacts: int
  """Number of tracked contact points (e.g. feet)."""
  num_joints: int
  """Actuated joints reported by the torque head."""
  num_reward_terms: int
  """Decomposed reward terms predicted alongside the total reward."""
  design_dim: int


def contact_feature_dim(num_contacts: int) -> int:
  """Dimension of :func:`contact_feature_vector` output."""
  return num_contacts * 6


def contact_feature_vector(
  found: torch.Tensor,
  force: torch.Tensor,
  air_time: torch.Tensor,
  contact_time: torch.Tensor,
) -> torch.Tensor:
  """Privileged contact features: [found, log-forces, squashed air/contact time].

  Shapes: found/air_time/contact_time (..., F), force (..., F, 3). Output
  (..., F*6). Forces are sign-log scaled (impacts span orders of magnitude);
  times are tanh-squashed with a 0.5 s scale.
  """
  force_flat = force.flatten(start_dim=-2)
  force_feat = torch.sign(force_flat) * torch.log1p(force_flat.abs())
  return torch.cat(
    [
      found,
      force_feat,
      torch.tanh(air_time / 0.5),
      torch.tanh(contact_time / 0.5),
    ],
    dim=-1,
  )


def gumbel_sigmoid(logits: torch.Tensor, tau: float, hard: bool = True) -> torch.Tensor:
  """Straight-through Gumbel-sigmoid over independent Bernoulli logits."""
  u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
  gumbel_noise = torch.log(u) - torch.log1p(-u)
  soft = torch.sigmoid((logits + gumbel_noise) / tau)
  if not hard:
    return soft
  return soft + ((soft > 0.5).float() - soft).detach()


class ConditionedMLP(nn.Module):
  """MLP whose hidden layers are FiLM-modulated by a conditioning vector.

  Each hidden layer computes ``mish(ln(W x) * (1 + scale) + shift)`` with
  (scale, shift) generated from the conditioning vector. With ``cond_dim=0``
  the module reduces to a plain LayerNorm-Mish MLP.
  """

  def __init__(
    self,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    cond_dim: int = 0,
  ) -> None:
    super().__init__()
    self.cond_dim = cond_dim
    self.layers = nn.ModuleList()
    self.norms = nn.ModuleList()
    self.films = nn.ModuleList()
    dim = in_dim
    for _ in range(num_layers):
      self.layers.append(nn.Linear(dim, hidden_dim))
      self.norms.append(nn.LayerNorm(hidden_dim))
      if cond_dim > 0:
        # Near-identity modulation at init (stable early training), but with
        # small nonzero weights so the conditioning pathway is active — FiLM
        # is the only route for design/mode information into the trunk.
        film = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.normal_(film.weight, std=0.02)
        nn.init.zeros_(film.bias)
        self.films.append(film)
      dim = hidden_dim
    self.act = nn.Mish()
    self.out = nn.Linear(dim, out_dim)

  def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
    if (cond is None) != (self.cond_dim == 0):
      raise ValueError("cond must be provided iff cond_dim > 0.")
    for i, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
      h = norm(layer(x))
      if cond is not None:
        scale, shift = self.films[i](cond).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
      x = self.act(h)
    return self.out(x)


class DynamicsMember(nn.Module):
  """One ensemble member: contact-gated latent dynamics + reward head."""

  def __init__(self, cfg: WorldModelCfg, dims: WorldModelDims) -> None:
    super().__init__()
    self._cfg = cfg
    in_dim = cfg.latent_dim + dims.action_dim + dims.command_dim
    self.predicts_contact = cfg.contact_heads or cfg.contact_gating
    if self.predicts_contact:
      self.next_contact = ConditionedMLP(
        in_dim, cfg.hidden_dim, dims.num_contacts, 2, cfg.design_embed_dim
      )
    trunk_cond = cfg.design_embed_dim + (
      cfg.mode_embed_dim if cfg.contact_gating else 0
    )
    self.trunk = ConditionedMLP(
      in_dim, cfg.hidden_dim, cfg.latent_dim, cfg.num_layers, trunk_cond
    )
    self.simnorm = SimNorm(cfg.simnorm_group_size)
    self.reward = ConditionedMLP(
      in_dim,
      cfg.hidden_dim,
      1 + dims.num_reward_terms,
      cfg.num_layers,
      cfg.design_embed_dim,
    )


class WorldModel(nn.Module):
  """Contact-enhanced, design-conditioned latent world model."""

  def __init__(self, cfg: WorldModelCfg, dims: WorldModelDims) -> None:
    super().__init__()
    if cfg.conditioning != "film":
      raise NotImplementedError(
        f"conditioning={cfg.conditioning!r}; the hypernetwork ablation arm "
        "is part of the Phase-4 mechanism study."
      )
    self.cfg = cfg
    self.dims = dims

    enc_in = dims.obs_dim
    if cfg.privileged_encoder:
      enc_in += contact_feature_dim(dims.num_contacts)
    self.encoder = nn.Sequential(
      ConditionedMLP(enc_in, cfg.hidden_dim, cfg.latent_dim, cfg.num_layers),
      SimNorm(cfg.simnorm_group_size),
    )
    self.target_encoder = copy.deepcopy(self.encoder)
    self.target_encoder.requires_grad_(False)

    self.design_encoder = nn.Sequential(
      nn.Linear(dims.design_dim, cfg.design_embed_dim),
      nn.Mish(),
      nn.Linear(cfg.design_embed_dim, cfg.design_embed_dim),
    )
    if cfg.contact_gating:
      self.mode_embedding = nn.Parameter(
        torch.randn(dims.num_contacts, cfg.mode_embed_dim) * 0.02
      )

    self.members = nn.ModuleList(
      DynamicsMember(cfg, dims) for _ in range(cfg.ensemble_size)
    )

    e_dim = cfg.design_embed_dim
    h, latent = cfg.hidden_dim, cfg.latent_dim
    if cfg.contact_heads:
      self.contact_head = ConditionedMLP(latent, h, dims.num_contacts * 4, 2, e_dim)
    self.termination_head = ConditionedMLP(latent, h, 1, 2, e_dim)
    self.torque_head = ConditionedMLP(
      latent + dims.action_dim, h, dims.num_joints, 2, e_dim
    )
    self.obs_head = ConditionedMLP(latent, h, dims.obs_dim, 2, e_dim)

  # -- Encoding ---------------------------------------------------------------

  def encode(
    self, obs: torch.Tensor, contact_vec: torch.Tensor | None = None
  ) -> torch.Tensor:
    if self.cfg.privileged_encoder:
      if contact_vec is None:
        raise ValueError("privileged_encoder=True requires contact features.")
      obs = torch.cat([obs, contact_vec], dim=-1)
    return self.encoder(obs)

  @torch.no_grad()
  def encode_target(
    self, obs: torch.Tensor, contact_vec: torch.Tensor | None = None
  ) -> torch.Tensor:
    if self.cfg.privileged_encoder:
      assert contact_vec is not None
      obs = torch.cat([obs, contact_vec], dim=-1)
    return self.target_encoder(obs)

  def embed_design(self, theta_normalized: torch.Tensor) -> torch.Tensor:
    return self.design_encoder(theta_normalized)

  @torch.no_grad()
  def update_target(self, tau: float) -> None:
    for p, tp in zip(
      self.encoder.parameters(), self.target_encoder.parameters(), strict=True
    ):
      tp.lerp_(p, 1.0 - tau)

  # -- Dynamics ---------------------------------------------------------------

  def step(
    self,
    member: int,
    z: torch.Tensor,
    action: torch.Tensor,
    command: torch.Tensor | None,
    e: torch.Tensor,
    true_next_contact: torch.Tensor | None = None,
    sample_mode: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One latent step of ensemble member ``member``.

    Returns ``(z_next, next_contact_logits)``. The mode code that gates the
    trunk is ``true_next_contact`` when given (teacher forcing during
    training), otherwise the head's own prediction (Gumbel-sampled when
    ``sample_mode`` for scheduled sampling, hard-thresholded at eval).
    """
    m = self.members[member]
    assert isinstance(m, DynamicsMember)
    x = _cat(z, action, command)
    logits = m.next_contact(x, e) if m.predicts_contact else None

    cond = e
    if self.cfg.contact_gating:
      assert logits is not None
      if true_next_contact is not None:
        mode = true_next_contact
      elif sample_mode:
        mode = gumbel_sigmoid(logits, self.cfg.gumbel_tau)
      elif self.training:
        # Soft-through prediction keeps gradients into the contact logits.
        mode = torch.sigmoid(logits)
      else:
        mode = (logits > 0).float()
      cond = torch.cat([e, mode @ self.mode_embedding], dim=-1)

    z_next = m.simnorm(m.trunk(x, cond))
    return z_next, logits

  # -- Heads ------------------------------------------------------------------

  def reward(
    self,
    member: int,
    z: torch.Tensor,
    action: torch.Tensor,
    command: torch.Tensor | None,
    e: torch.Tensor,
  ) -> torch.Tensor:
    """(..., 1 + num_reward_terms): total reward then decomposed terms."""
    m = self.members[member]
    assert isinstance(m, DynamicsMember)
    return m.reward(_cat(z, action, command), e)

  def contact(
    self, z: torch.Tensor, e: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Current-contact prediction: (found logits (..., F), forces (..., F, 3))."""
    out = self.contact_head(z, e)
    f = self.dims.num_contacts
    return out[..., :f], out[..., f:].unflatten(-1, (f, 3))

  def termination(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return self.termination_head(z, e).squeeze(-1)

  def torque(
    self, z: torch.Tensor, action: torch.Tensor, e: torch.Tensor
  ) -> torch.Tensor:
    return self.torque_head(torch.cat([z, action], dim=-1), e)

  def predict_obs(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return self.obs_head(z, e)


def _cat(
  z: torch.Tensor, action: torch.Tensor, command: torch.Tensor | None
) -> torch.Tensor:
  parts = [z, action] if command is None else [z, action, command]
  return torch.cat(parts, dim=-1)
