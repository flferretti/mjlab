"""Tests for the world-model networks (shapes, gating, conditioning)."""

import pytest
import torch

from mjlab.world_model import WorldModel, WorldModelCfg, WorldModelDims
from mjlab.world_model.networks import contact_feature_vector, gumbel_sigmoid
from mjlab.world_model.simnorm import SimNorm

DIMS = WorldModelDims(
  obs_dim=20,
  action_dim=6,
  command_dim=3,
  num_contacts=2,
  num_joints=6,
  num_reward_terms=2,
  design_dim=3,
)


def small_cfg(**overrides) -> WorldModelCfg:
  defaults: dict = dict(
    latent_dim=64,
    hidden_dim=32,
    num_layers=2,
    design_embed_dim=8,
    mode_embed_dim=4,
    ensemble_size=2,
    horizon=4,
  )
  defaults.update(overrides)
  return WorldModelCfg(**defaults)


@pytest.fixture
def batch():
  torch.manual_seed(0)
  return {
    "obs": torch.randn(5, DIMS.obs_dim),
    "action": torch.randn(5, DIMS.action_dim),
    "command": torch.randn(5, DIMS.command_dim),
    "theta": torch.rand(5, DIMS.design_dim),
  }


def test_simnorm_groups_sum_to_one():
  x = torch.randn(3, 16)
  out = SimNorm(group_size=8)(x)
  sums = out.view(3, 2, 8).sum(dim=-1)
  assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_simnorm_rejects_indivisible_dim():
  with pytest.raises(ValueError, match="divisible"):
    SimNorm(group_size=8)(torch.randn(2, 20))


def test_world_model_shapes(batch):
  model = WorldModel(small_cfg(), DIMS)
  e = model.embed_design(batch["theta"])
  z = model.encode(batch["obs"])
  assert z.shape == (5, 64)
  z_next, logits = model.step(0, z, batch["action"], batch["command"], e)
  assert z_next.shape == (5, 64)
  assert logits is not None and logits.shape == (5, DIMS.num_contacts)
  assert model.reward(0, z, batch["action"], batch["command"], e).shape == (
    5,
    1 + DIMS.num_reward_terms,
  )
  found_logits, forces = model.contact(z, e)
  assert found_logits.shape == (5, DIMS.num_contacts)
  assert forces.shape == (5, DIMS.num_contacts, 3)
  assert model.termination(z, e).shape == (5,)
  assert model.torque(z, batch["action"], e).shape == (5, DIMS.num_joints)
  assert model.predict_obs(z, e).shape == (5, DIMS.obs_dim)


def test_contact_blind_variant_has_no_contact_paths(batch):
  model = WorldModel(small_cfg(contact_heads=False, contact_gating=False), DIMS)
  e = model.embed_design(batch["theta"])
  z = model.encode(batch["obs"])
  z_next, logits = model.step(0, z, batch["action"], batch["command"], e)
  assert logits is None
  assert z_next.shape == (5, 64)
  assert not hasattr(model, "contact_head")


def test_design_conditioning_changes_dynamics(batch):
  torch.manual_seed(1)
  model = WorldModel(small_cfg(), DIMS)
  z = model.encode(batch["obs"])
  e1 = model.embed_design(torch.zeros(5, DIMS.design_dim))
  e2 = model.embed_design(torch.ones(5, DIMS.design_dim))
  model.eval()
  z1, _ = model.step(0, z, batch["action"], batch["command"], e1)
  z2, _ = model.step(0, z, batch["action"], batch["command"], e2)
  assert not torch.allclose(z1, z2)


def test_teacher_forced_mode_changes_dynamics(batch):
  torch.manual_seed(2)
  model = WorldModel(small_cfg(), DIMS)
  model.eval()
  e = model.embed_design(batch["theta"])
  z = model.encode(batch["obs"])
  on = torch.ones(5, DIMS.num_contacts)
  off = torch.zeros(5, DIMS.num_contacts)
  z_on, _ = model.step(0, z, batch["action"], batch["command"], e, on)
  z_off, _ = model.step(0, z, batch["action"], batch["command"], e, off)
  assert not torch.allclose(z_on, z_off)


def test_ema_target_update_moves_target(batch):
  model = WorldModel(small_cfg(), DIMS)
  before = [p.clone() for p in model.target_encoder.parameters()]
  with torch.no_grad():
    for p in model.encoder.parameters():
      p.add_(1.0)
  model.update_target(tau=0.5)
  moved = any(
    not torch.allclose(b, a)
    for b, a in zip(before, model.target_encoder.parameters(), strict=True)
  )
  assert moved


def test_gumbel_sigmoid_straight_through():
  logits = torch.zeros(64, 2, requires_grad=True)
  out = gumbel_sigmoid(logits, tau=1.0)
  assert set(out.detach().unique().tolist()) <= {0.0, 1.0}
  out.sum().backward()
  assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_contact_feature_vector_shape():
  found = torch.ones(5, 2)
  force = torch.randn(5, 2, 3)
  air = torch.rand(5, 2)
  ct = torch.rand(5, 2)
  feat = contact_feature_vector(found, force, air, ct)
  assert feat.shape == (5, 12)
  assert torch.isfinite(feat).all()


def test_privileged_encoder_requires_contact(batch):
  model = WorldModel(small_cfg(privileged_encoder=True), DIMS)
  with pytest.raises(ValueError, match="contact"):
    model.encode(batch["obs"])
  feat = contact_feature_vector(
    torch.ones(5, 2), torch.randn(5, 2, 3), torch.rand(5, 2), torch.rand(5, 2)
  )
  assert model.encode(batch["obs"], feat).shape == (5, 64)
