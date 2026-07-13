"""Contract tests for the world-model co-design backend (mocked evaluator)."""

import numpy as np
import pytest
import torch

from mjlab.world_model.codesign_backend import WorldModelBackend
from mjlab.world_model.config import WorldModelBackendCfg
from mjlab.world_model.imagination import DesignScores

JOINT_ORDER = ["l_a", "r_a", "l_b", "r_b"]
N_JOINTS = len(JOINT_ORDER)


class FakeEvaluator:
  """Deterministic evaluator: performance = mean tau, biased by +1."""

  def __init__(self, uncertainty=None):
    self.uncertainty = uncertainty
    self.calls = 0

  def evaluate_joint_tau(self, tau_LJ, joint_order, n_seeds):
    self.calls += 1
    n = tau_LJ.shape[0]
    perf = tau_LJ.float().mean(dim=1) + 1.0
    unc = self.uncertainty[:n] if self.uncertainty is not None else torch.zeros(n)
    return DesignScores(
      performance=perf.cpu(),
      rms_torque=0.25 * tau_LJ.float().cpu(),
      uncertainty=unc,
    )


class FakeVerifier:
  """True-sim stand-in: performance = mean tau (unbiased)."""

  def __init__(self):
    self.last_rms_LJ = None
    self.evaluated = []

  def evaluate(self, tau_LJ, n_seeds):
    self.evaluated.append(tau_LJ.shape[0])
    self.last_rms_LJ = 0.5 * tau_LJ.numpy()
    return tau_LJ.float().mean(dim=1).numpy().astype(np.float64)


def make_backend(verifier=None, uncertainty=None, **cfg_kwargs):
  cfg = WorldModelBackendCfg(**cfg_kwargs)
  return WorldModelBackend(
    evaluator=FakeEvaluator(uncertainty),
    joint_order=JOINT_ORDER,
    rollout_steps=10,
    verifier=verifier,
    cfg=cfg,
  )


def test_evaluate_contract_without_verifier():
  backend = make_backend(verify_topk=0.0)
  tau = torch.rand(8, N_JOINTS) * 50 + 10
  scores = backend.evaluate(tau, n_seeds=2)
  assert isinstance(scores, np.ndarray)
  assert scores.shape == (8,) and scores.dtype == np.float64
  assert backend.last_rms_LJ is not None
  assert backend.last_rms_LJ.shape == (8, N_JOINTS)


def test_verifier_overwrites_topk_scores_and_rms():
  verifier = FakeVerifier()
  backend = make_backend(verifier=verifier, verify_topk=0.25, calibrate=False)
  tau = torch.arange(8 * N_JOINTS).reshape(8, N_JOINTS).float()
  scores = backend.evaluate(tau, n_seeds=1)
  # ceil(0.25 * 8) = 2 designs verified: the top-2 predicted (rows 6, 7).
  assert verifier.evaluated == [2]
  assert scores[7] == pytest.approx(float(tau[7].mean()))  # true, not +1
  assert scores[6] == pytest.approx(float(tau[6].mean()))
  assert scores[0] == pytest.approx(float(tau[0].mean()) + 1.0)  # prediction
  assert backend.last_rms_LJ is not None
  np.testing.assert_allclose(backend.last_rms_LJ[7], 0.5 * tau[7].numpy())
  np.testing.assert_allclose(backend.last_rms_LJ[0], 0.25 * tau[0].numpy())
  assert backend.verified_fraction == pytest.approx(2 / 8)


def test_high_uncertainty_triggers_verification():
  verifier = FakeVerifier()
  unc = torch.zeros(8)
  unc[2] = 100.0  # far above the running percentile
  backend = make_backend(verifier=verifier, uncertainty=unc, verify_topk=0.125)
  # Seed the uncertainty history so the percentile threshold is defined.
  backend._uncertainty_history = [0.01] * 30
  tau = torch.arange(8 * N_JOINTS).reshape(8, N_JOINTS).float()
  backend.evaluate(tau, n_seeds=1)
  # Top-1 (row 7) plus the uncertain row 2.
  assert verifier.evaluated == [2]


def test_affine_calibration_debiases_predictions():
  verifier = FakeVerifier()
  backend = make_backend(
    verifier=verifier, verify_topk=0.5, calibrate=True, min_calibration_pairs=4
  )
  tau = torch.rand(8, N_JOINTS) * 50 + 10
  backend.evaluate(tau, n_seeds=1)  # Gathers 4 calibration pairs.
  assert len(backend._calibration) >= 4
  # With calibration fit on (pred = true + 1) pairs, unverified predictions
  # should now match the true values.
  backend.cfg.verify_topk = 0.0
  backend.verifier = None
  scores = backend.evaluate(tau, n_seeds=1)
  np.testing.assert_allclose(
    scores, tau.float().mean(dim=1).numpy(), rtol=1e-4, atol=1e-4
  )
